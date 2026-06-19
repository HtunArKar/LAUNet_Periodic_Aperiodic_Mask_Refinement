# import necessary packages
from Model.LAUNET_ConFormer import LAUNET, pad_to_multiple

from utils.harmonic_residual_mask import hrps

from utils.dataset import CustomDataset
from utils.se_utils import power_compress, power_uncompress


from torch.nn.parallel import DistributedDataParallel as DDP
from matplotlib import pyplot as plt
from IPython.display import Audio, display
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch import nn
from socket import gethostname
from collections import defaultdict
import noisereduce as nr
import torch.distributed as dist
import numpy as np
import torchaudio
import torch
import argparse
import json
import time
import os




def forward_step(model, clean_batch, noisy_batch, device, config):

    num_layers = len(config.num_channels) - 2

    dtype = clean_batch.dtype
    length = clean_batch.shape[-1]

    window = torch.hann_window(config.n_fft, device=device, dtype=dtype)
 

    # get spectrogram with short time fourier transform
    noisy_spec = torch.stft(noisy_batch, config.n_fft, config.hop, win_length=config.n_fft,
                            window=window, center=True, return_complex=True) # shape = [B?, F, T]
    clean_spec = torch.stft(clean_batch, config.n_fft, config.hop, win_length=config.n_fft,
                            window=window, center=True, return_complex=True) # shape = [B?, F, T]

    # get original dim
    B, Freq, T = clean_spec.shape
    clean_orig_spec = clean_spec
    
    
    # get harmonic masks
    noisy_harmonic_mask, _ = hrps(noisy_spec, Fs=config.sampling_rate, N=config.n_fft, H=config.hop,
                                  L_h=config.L_h_sec, L_p=config.L_p_Hz, beta=config.beta) # [B?, F, T]

    clean_harmonic_mask, _ = hrps(clean_spec, Fs=config.sampling_rate, N=config.n_fft, H=config.hop,
                                  L_h=config.L_h_sec, L_p=config.L_p_Hz, beta=config.beta)  
                                  
    
    # get clea real and imag target for loss calculation
    clean_spec_compress = power_compress(clean_spec) # shape = [B?, 2, F, T]

    clean_real = clean_spec_compress[:, 0, :, :].unsqueeze(1) # shape = [B?, F, T] => [B?, 1, F, T]
    clean_imag = clean_spec_compress[:, 1, :, :].unsqueeze(1) # shape = [B?, F, T]

    # input preparation
    inputs = power_compress(noisy_spec) # [B?, C, F, T]
    inputs = inputs.permute(0, 1, 3, 2)  # [B?, C, T, F]

    harmonic_mask = noisy_harmonic_mask.permute(0, 2, 1).unsqueeze(1) # [B?, C, T, F]

    inputs = pad_to_multiple(inputs, mode=config.mode, multiple=2 ** num_layers)
    harmonic_mask = pad_to_multiple(harmonic_mask, mode=config.mode, multiple=2 ** num_layers)
    
    # harmonic_mask = harmonic_mask.float()

    # forward pass into the model
    est_spec = model(inputs, harmonic_mask) # [B?, 2, T, F]

    # unpack the output
    est_spec = est_spec[:, :, :T, :Freq]

    est_real = est_spec[:, 0:1, :, :].permute(0, 1, 3, 2) # [B?, 1, F, T]
    est_imag = est_spec[:, 1:2, :, :].permute(0, 1, 3, 2) # [B?, 1, F, T]

    # reconstruct the audio from estimated real and imaginary part.
    spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)

    if not torch.is_complex(spec_uncompress):

        spec_uncompress = torch.view_as_complex(spec_uncompress)

    est_wav = torch.istft(spec_uncompress, config.n_fft, config.hop, win_length=config.n_fft,
                        window=window, center=True, length=length)

    # get clean harmonic and est harmonic for loss calculation
    clean_harmonic = clean_harmonic_mask * clean_spec
    est_harmonic = clean_harmonic_mask * spec_uncompress
    
    
    """
    ##########################################################################################
    
    beta = -5.9
    
    alpha = 6
    
    
    # adding harmonic loss
    # complex spectral error
    spec_error = clean_orig_spec - spec_uncompress
    
    # calculate harmonic weight from clean magnitude
    mag = torch.abs(clean_orig_spec).clamp_min(1e-8)
    log_mag = torch.log(mag)
    
    beta = torch.tensor(beta, device=device, dtype=log_mag.dtype)
    
    W = torch.maximum(log_mag, beta) + alpha
    
    # normalize per-utterance (per sample) over (F,T)
    W_min = W.amin(dim=(-2, -1), keepdim=True)
    W_max = W.amax(dim=(-2, -1), keepdim=True)
    W_norm = (W - W_min) / (W_max - W_min + 1e-8)

    
    # weigth complex error
    h_spec_error = W_norm * spec_error
    
    # get error in time domain
    error_harmonic = torch.istft(h_spec_error, config.n_fft, config.hop, window=window, center=True)
    
    
    ##########################################################################################
    """
    
    error_harmonic = None


    return {
        "est_real": est_real,
        "est_imag": est_imag,
        "clean_real": clean_real,
        "clean_imag": clean_imag,
        "est_wav": est_wav,
        "clean_wav": clean_batch,
        "est_harmonic": est_harmonic,
        "clean_harmonic": clean_harmonic,
        "harmonic_error": error_harmonic
    }



def SISNRLoss(target, estimate, epsilon=1e-5):

    # Standard SI-SNR calculation
    target = target - torch.mean(target, dim=-1, keepdim=True)
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)

    # Scale factor
    s_target = torch.sum(estimate * target, dim=-1, keepdim=True) * target / \
                (torch.sum(target ** 2, dim=-1, keepdim=True) + epsilon)

    # Noise/error
    e_noise = estimate - s_target

    # Standard SI-SNR (per sample)
    si_snr = torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + epsilon)
    si_snr_db = 10 * torch.log10(si_snr + epsilon)

    return -torch.mean(si_snr_db)




def calculate_loss(outputs, alpha_1=0.2, alpha_2=0.5, alpha_3=0.1, alpha_4=0.5):

    
    # calculate harmonic loss
    harmonic_loss = torch.mean(torch.abs(outputs['est_harmonic'] - outputs['clean_harmonic']))
    
    
    # calculate harmonic loss in time domain
    # harmonic_loss = torch.mean(outputs["harmonic_error"] * outputs["harmonic_error"])


    # calculate harmonic complex loss
    loss_ri = F.mse_loss(
        outputs['est_real'], outputs['clean_real']
    ) + F.mse_loss(outputs['est_imag'], outputs['clean_imag'])

    # calculate residual loss
    si_snr_loss = SISNRLoss(outputs["clean_wav"], outputs["est_wav"])

    # calculate time loss
    time_loss = torch.mean(
        torch.abs(outputs["est_wav"] - outputs["clean_wav"])
    )

    # combine three losses
    loss = (alpha_1 * harmonic_loss) + (alpha_2 * loss_ri) +  (alpha_3 * si_snr_loss) + (alpha_4 * time_loss)
    
    # loss = (alpha_2 * loss_ri) +  (alpha_3 * si_snr_loss) + (alpha_4 * time_loss)

    # return total generator loss
    return loss



def train(model, trainDataLoader, device, optimizer, log_interval, config=None, alpha_1=0.2, alpha_2=0.5, alpha_3=0.1, alpha_4=0.5):

    # set the model to train mode
    model.train()

    # initialize variable to store loss
    totalTrainLoss = 0

    start = time.time()

    # loop over batch
    for (i, (clean, noisy, gt_files, length)) in enumerate(trainDataLoader):

        # set the input to device
        (clean, noisy) = (clean.to(device), noisy.to(device))

        # forward pass
        outputs = forward_step(model, clean, noisy, device, config=config)


        # Calculate Generator loss
        loss = calculate_loss(outputs, alpha_1, alpha_2, alpha_3, alpha_4)


        # backward pass for generator
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update loss
        totalTrainLoss += loss.item()


        if (i + 1) % log_interval == 0 and device == 0:

            print(f"[INFO] Processing Train Batch: {i + 1}/{len(trainDataLoader)}, train_batch_gen_loss: {(totalTrainLoss / (i + 1)):.4f}, \
            time_taken: {((time.time() - start) / 60):.4f} mins.")

            start = time.time()


    return totalTrainLoss


def validation(model, valDataLoader, device, log_interval, config=None, alpha_1=0.2, alpha_2=0.5, alpha_3=0.1, alpha_4=0.5):

    # validation stage
    # set the model to validation mode
    model.eval()

    with torch.no_grad():

        totalValLoss = 0

        start = time.time()

        # loop over validation data loader
        for (i, (clean, noisy, gt_files, length)) in enumerate(valDataLoader):

            # set the inputs to device
            (clean, noisy) = (clean.to(device), noisy.to(device))

            # forward pass
            outputs = forward_step(model, clean, noisy, device, config=config)

            # Calculate Generator loss
            loss = calculate_loss(outputs, alpha_1, alpha_2, alpha_3, alpha_4)

            # update loss
            totalValLoss += loss.item()

            if (i + 1) % log_interval == 0 and device == 0:

                print(f"[INFO] Processing Val Batch: {i + 1}/{len(valDataLoader)}, train_batch_gen_loss: {(totalValLoss / (i + 1)):.4f}, \
                time_taken: {((time.time() - start) / 60):.4f} mins.")

                start = time.time()

    return totalValLoss



def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        


def main():

    # create Argumentparser
    parser = argparse.ArgumentParser(description="Training SE Plus ASR Model.")

    # add arguments
    parser.add_argument("--clean_path", type=str, help="Path to clean files.")
    parser.add_argument("--noisy_path", type=str, help="Path to noisy files.")
    parser.add_argument("--loss_weights", nargs=4, type=float, default=[0.2, 0.5, 0.1, 0.5],
                        help="weights of complex loss and time loss")
    parser.add_argument("--save_every", type=int, help="Save for every defined epoch")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="disable CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--log_interval", type=int, default=10, metavar='N', help="how many batches to wait before logging.")
    parser.add_argument("--config_path", type=str, help="Path to configuration file.")
    parser.add_argument("--base_output_path", type=str, help="Path to store losses.")
    parser.add_argument("--checkpoint_path", type=str, help="Path to store checkpoint model.")

    args = parser.parse_args()

    # check the cuda is used or not
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # set random value
    torch.manual_seed(args.seed)

    # load configuration file
    with open(args.config_path) as file:
        config = json.load(file)

    h = AttrDict(config)

    # set train and validation keyword arguments
    train_kwargs = {'batch_size': h.batch_size}
    val_kwargs = {'batch_size': h.batch_size}

    if use_cuda:

        cuda_kwargs = {'num_workers': int(os.environ["SLURM_CPUS_PER_TASK"]),
                       'pin_memory': True,
                       'shuffle': True,
                       'drop_last': True}

        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)

    # get world_size, rank and gpus_per_node
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])

    assert gpus_per_node == torch.cuda.device_count()

    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    setup(rank, world_size)
    if rank == 0: print(f"Gruop initialized? {dist.is_initialized()}", flush=True)

    # get local rank and set device
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    # Prepare Datasets and DataLoaders
    clean_files = [os.path.join(args.clean_path, path) for path in sorted(os.listdir(args.clean_path))]
    noisy_files = [os.path.join(args.noisy_path, path) for path in sorted(os.listdir(args.noisy_path))]
    
    train_clean_files, val_clean_files, train_noisy_files, val_noisy_files = train_test_split(clean_files, noisy_files, test_size=0.2, random_state=42)
    
    
    if rank == 0:
        print(f"[INFO] Number of training samples: {len(train_clean_files)}.")
        print(f"[INFO] Number of validation samples: {len(val_clean_files)}.")
    
    # check whether the split method is correct or not
    idx = np.random.randint(0, len(train_clean_files))
    
    text = "File names must be the same."
    assert train_clean_files[idx].split("/")[-1].split(".")[0] == train_noisy_files[idx].split("/")[-1].split("-")[0], text
    
    # create train and validation datasets and dataloader                                
    trainDataset = CustomDataset(clean_files=train_clean_files, noisy_files=train_noisy_files,
                                sampling_rate=h.sampling_rate, cut_len=h.cut_len)
                                 
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainDataset, num_replicas=world_size, rank=rank, shuffle=True)
                                                
                                                
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=h.batch_size, sampler=train_sampler, \
                                               num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True, drop_last=True)
    
    valDataset = CustomDataset(clean_files=val_clean_files, noisy_files=val_noisy_files,
                            sampling_rate=h.sampling_rate, cut_len=h.cut_len)

    valDataLoader = DataLoader(valDataset, batch_size=h.batch_size, num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), \
                               pin_memory=True, drop_last=True)


    # load the model and set to multi-gpus training mode
    
    model = LAUNET(num_channels=h.num_channels, attn_layers=h.attn_layers, conf_num_layers=h.conf_num_layers, 
                   mode=h.mode, conv_kernel_size=h.conv_kernel_size, heads=h.head, ff_mult=h.ff_mult, 
                   expansion_factor=h.expansion_factor, attn_dropout=h.attn_dropout, 
                   ff_dropout=h.ff_dropout, conv_dropout=h.conv_dropout).to(local_rank)
      

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # intitalize optimizer and scheduler
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=h.init_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=h.step_size, gamma=h.gamma)
    

    # initialize a dictionary to store loss
    history = {"train_epoch_loss": [],
               "val_epoch_loss": []}

    start = time.time()

    # loop over epochs
    for epoch in range(0, h.num_epochs):

        avgTrainLoss = 0.0
        avgValLoss = 0.0
        
           
        # training stage
        totalTrainLoss = train(ddp_model, trainDataLoader, local_rank, optimizer, args.log_interval, config=h, 
                               alpha_1=args.loss_weights[0], alpha_2=args.loss_weights[1], alpha_3=args.loss_weights[2], alpha_4=args.loss_weights[3])

        
        if rank == 0:
        
            totalValLoss = validation(ddp_model, valDataLoader, local_rank, args.log_interval, config=h, 
                                      alpha_1=args.loss_weights[0], alpha_2=args.loss_weights[1], alpha_3=args.loss_weights[2], alpha_4=args.loss_weights[3])
            

            # calculate epoch train and validation loss, average
            avgTrainLoss = totalTrainLoss / len(trainDataLoader)
            avgValLoss = totalValLoss / len(valDataLoader)

            # update result
            history['train_epoch_loss'].append(avgTrainLoss)
            history['val_epoch_loss'].append(avgValLoss)

            print(f"[INFO] Processing Epoch: {epoch + 1}/{h.num_epochs}, train_epoch_loss: {avgTrainLoss:.4f}, \
            val_epoch_loss: {avgValLoss:.4f}, time_taken: {((time.time() - start) / 60):.4f} mins.\n")

            start = time.time()


        if (epoch + 1) % args.save_every == 0 and rank == 0:

            checkpoint_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pth")

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "current_epoch": epoch + 1,
                "current_loss": history
            }, checkpoint_path)

            current_output_path = os.path.join(args.base_output_path, f"epoch_{epoch + 1}.png")

            plt.figure()
            plt.plot(np.arange(len(history['train_epoch_loss'])), history['train_epoch_loss'], label='Train Loss')
            plt.plot(np.arange(len(history['val_epoch_loss'])), history['val_epoch_loss'], label='Validation Loss')
            plt.title("Train and validation loss")
            plt.xlabel("Number of epochs")
            plt.ylabel("Loss")
            plt.legend(loc="upper right")
            plt.savefig(current_output_path)

        # update scheduler
        scheduler.step()
        

    # after the training destroy the process group
    dist.destroy_process_group()

# main function
if __name__ == '__main__':
    main()

