# import necessary packages
from Model.LAUNET_ConFormer import LAUNET, pad_to_multiple

from utils.harmonic_residual_mask import hrps

from utils.calculate_objectives_modified import tokenize_text, audio_to_text_thonburian_whisper, calculate_objectives, calculate_psnr
from utils.compute_metrics import compute_metrics

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import defaultdict
from jiwer import process_words, process_characters
from pythainlp.tokenize import word_tokenize
from pesq import pesq
from pystoi import stoi
from torch import nn
import argparse
import numpy as np
import torchaudio
import librosa
import torch
import string
import time
import re
import json
import math
import os



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def power_compress(x):
    mag = torch.abs(x)
    phase = torch.angle(x)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)


def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)



def perform_denoising(noisy_audio, clean_audio, model, device, config, cut_len=16*16000):


    num_layers = len(config.num_channels) - 2

    # set the input audio and model to device
    noisy = noisy_audio.to(device)
    clean = clean_audio.to(device)
    model = model.to(device)

    # perform padding
    length = noisy.size(-1)
    #frame_num = int(np.ceil(length / config.hop))
    #padded_len = frame_num * config.hop
    #padding_len = padded_len - length

    #noisy  = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)

    window = torch.hann_window(config.n_fft, device=device, dtype=noisy_audio.dtype)

    # get spectrogram with short time fourier transform
    noisy_spec = torch.stft(noisy, config.n_fft, config.hop, win_length=config.n_fft,
                            window=window, center=True, return_complex=True) # shape = [B?, F, T]
                            
    clean_spec = torch.stft(clean, config.n_fft, config.hop, win_length=config.n_fft,
                            window=window, center=True, return_complex=True) # shape = [B?, F, T]

    # get original dim
    B, Freq, T = noisy_spec.shape
    
    
    noisy_harmonic_mask, _ = hrps(noisy_spec, Fs=config.sampling_rate, N=config.n_fft, H=config.hop,
                                  L_h=config.L_h_sec, L_p=config.L_p_Hz, beta=config.beta) # [B?, F, T]
                               

                                  
    # input preparation
    inputs = power_compress(noisy_spec) # [B?, C, F, T]
    inputs = inputs.permute(0, 1, 3, 2)  # [B?, C, T, F]
    
    harmonic_mask = noisy_harmonic_mask.permute(0, 2, 1).unsqueeze(1) # [B?, C, T, F]


    inputs = pad_to_multiple(inputs, mode=config.mode, multiple=2 ** num_layers)
    harmonic_mask = pad_to_multiple(harmonic_mask, mode=config.mode, multiple=2 ** num_layers)
    
    # harmonic_mask = harmonic_mask.float()

    # forward pass into the model
    with torch.no_grad():
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

    est_audio = torch.flatten(est_wav)[:length].cpu().detach().numpy()


    return est_audio



def main():

    # create Argumentparser
    parser = argparse.ArgumentParser(description="Training Mossformer Model.")

    # add arguments
    parser.add_argument("--clean_path", type=str, help="Path to clean files.")
    parser.add_argument("--noisy_path", type=str, help="Path to noisy files.")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="disable CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--log_interval", type=int, default=10, metavar='N', help="how many batches to wait before logging.")
    parser.add_argument("--config_path", type=str, help="Path to configuration file.")
    parser.add_argument("--se_model_path", type=str, help="Path to Pretrained SE model.")
    parser.add_argument("--asr_model_path", type=str, help="Path to T-Whisper ASR model.")
    parser.add_argument("--gt_path", type=str, help="Path to Ground Truth file.")

    args = parser.parse_args()

    # check the cuda is used or not
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    if use_cuda:
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[INFO] {device} is used")

    # set random value
    torch.manual_seed(args.seed)

    # load configuration file
    with open(args.config_path) as file:
        config = json.load(file)

    h = AttrDict(config)
    
    # Prepare Datasets and DataLoaders
    clean_files = [os.path.join(args.clean_path, path) for path in sorted(os.listdir(args.clean_path))]
    noisy_files = [os.path.join(args.noisy_path, path) for path in sorted(os.listdir(args.noisy_path))]
    
    # get the random sample
    idx = np.random.randint(0, len(noisy_files))
    
    clean_file = clean_files[idx]
    noisy_file = noisy_files[idx]
    
    assert clean_file.split("/")[-1].split(".")[0] == noisy_file.split("/")[-1].split("-")[0], "File names must be the same."
    
    
    # load pretrained SE Model   
    model = LAUNET(num_channels=h.num_channels, attn_layers=h.attn_layers, conf_num_layers=h.conf_num_layers,
                   mode=h.mode, conv_kernel_size=h.conv_kernel_size, heads=h.head, ff_mult=h.ff_mult,
                   expansion_factor=h.expansion_factor, attn_dropout=h.attn_dropout,
                   ff_dropout=h.ff_dropout, conv_dropout=h.conv_dropout)
                       
    state = torch.load(args.se_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state['model_state_dict'])

    # set se model to evaluation mode
    model.eval()
    model.to(device)
    
    
    # load Thonburian Whisper Model
    thonburian_processor = WhisperProcessor.from_pretrained(args.asr_model_path, language="thai", task="transcribe", fast_tokenizer=True)
    thonburian_model = WhisperForConditionalGeneration.from_pretrained(args.asr_model_path).to(device)
    
    # set asr model to evaluation mode
    thonburian_model.eval()
    
    thonburian_model.config.forced_decoder_ids = thonburian_processor.get_decoder_prompt_ids(language="th", task="transcribe")
    
    
    # get the ground truth texts
    with open(args.gt_path) as file:
    
        gt_texts = file.readlines()
    
    gt_data = defaultdict()
    
    # loop over all texts
    for gt_text in gt_texts:
    
        file_name = gt_text.strip().split()[0]
        text = " ".join(gt_text.strip().split()[1:])
    
        # store into new dictionary
        gt_data[file_name] = text
        
    
    # Evaluation
    # initialize a dictionary to store result
    result = defaultdict(int)
    result['psnr'] = list()
    
    metrics_total = 0
    
    # initialize a dictionary to store result
    thonburian_objectives = defaultdict(float)
    
    start = time.time()
    
    for i in range(0, len(noisy_files)):
    
        # get the current clean and noisy file paths
        clean_file = clean_files[i]
        noisy_file = noisy_files[i]
    
        # double check
        text = "Clean and Noisy Files must be the same"
        assert clean_file.split("/")[-1].split(".")[0] == noisy_file.split("/")[-1].split("-")[0], text
    
        # get ground truth text
        name = noisy_file.split("/")[-1].split("-")[0]
        gt_text = gt_data[name]
        
        
    
        # read clean and noisy wav
        clean_audio, clean_sr = torchaudio.load(clean_file)
        noisy_audio, noisy_sr = torchaudio.load(noisy_file)
    
    
        if clean_sr != h.sampling_rate:
    
            resampler = torchaudio.transforms.Resample(orig_freq=clean_sr, new_freq=h.sampling_rate)
            clean_audio = resampler(clean_audio)
            clean_sr = h.sampling_rate
    
        if noisy_sr != h.sampling_rate:
    
            resampler = torchaudio.transforms.Resample(orig_freq=noisy_sr, new_freq=h.sampling_rate)
            noisy_audio = resampler(noisy_audio)
            noisy_sr = h.sampling_rate

    
        assert noisy_sr == clean_sr == h.sampling_rate, "All sampling rate must be equal"
    
        # perform denoising
        with torch.no_grad():
            denoised_audio = perform_denoising(noisy_audio, clean_audio, model, device, h)
    
        # generate text from remixed audio
        text_thonburian_whisper = audio_to_text_thonburian_whisper(denoised_audio, thonburian_model, thonburian_processor, sr=16000, device=device)
        
        # calculate ASR objective scores
        asr_outputs = calculate_objectives(gt_text, text_thonburian_whisper)
        
        
        # update ASR objectives
        # ASR Word Level
        thonburian_objectives["total_errors_words"] += asr_outputs['total_error_words']
        thonburian_objectives["total_words"] += asr_outputs['total_words']
        thonburian_objectives["total_words_insertions"] += asr_outputs['words_insertions']
        thonburian_objectives["total_words_deletions"] += asr_outputs['words_deletions']
        thonburian_objectives["total_words_substitutions"] += asr_outputs['words_substitutions']
    
        # ASR Chars Level
        thonburian_objectives["total_errors_chars"] += asr_outputs['total_error_chars']
        thonburian_objectives["total_chars"] += asr_outputs['total_chars']
        thonburian_objectives["total_chars_insertions"] += asr_outputs['chars_insertions']
        thonburian_objectives["total_chars_deletions"] += asr_outputs['chars_deletions']
        thonburian_objectives["total_chars_substitutions"] += asr_outputs['chars_substitutions']
    
        # BLEU Score
        thonburian_objectives["blue_1"] += asr_outputs['blue_1']
        thonburian_objectives["blue_2"] += asr_outputs['blue_2']
        thonburian_objectives["blue_4"] += asr_outputs['blue_4']
        thonburian_objectives["count"] += 1
        
        
        # ensure audio is in the range of -1 to 1 for PESQ calculation
        # get the sequence length
        length = clean_audio.shape[-1]
    
        # get the numpy array and remove batch dim
        clean_audio = clean_audio.cpu().numpy().squeeze(0)
        # denoised_audio = denoised_audio.squeeze(0)
    
        clean_audio = clean_audio / np.max(np.abs(clean_audio))
        denoised_audio = denoised_audio / np.max(np.abs(denoised_audio))
    
        # calculate PESQ and STOI
        sr = clean_sr
        result['pesq_wb'] += pesq(sr, clean_audio, denoised_audio, 'wb') * length  # wide band
        result['pesq_nb'] += pesq(sr, clean_audio, denoised_audio, 'nb') * length  # narrow band
        result['stoi'] += stoi(clean_audio, denoised_audio, sr, extended=False) * length
        result['psnr'].append(calculate_psnr(clean_audio, denoised_audio, max_val=1.0))
        result['count'] += 1 * length
    
        # calculate CSIG, CBAK, COVL, segSNR, and STOI
        # pesq_mos, CSIG, CBAK, COVL, segSNR, STOI
    
        metrics = compute_metrics(clean_audio, denoised_audio, 16000, path=0)
    
        metrics_total += np.array(metrics)
    
        if (i + 1) % args.log_interval == 0:
        
            print(f"[INFO] Processed: {i + 1}/{len(noisy_files)}, time taken: {(time.time() - start) / 60:.4f} mins.")
    
            start = time.time()
            
            
    # Display the results
    for key in result:
        if key != 'count' and key != 'psnr':
            print('{} = {:.4f}'.format(key, result[key]/result['count']), end=", ")
    
        elif key == 'psnr':
            psnr = result['psnr']
    
            print('psnr = {:.4f}'.format(np.mean(psnr)))
    
    print("\n\n[INFO] CSIG, CBAK, COVL, segSNR, and STOI")
    metrics_avg = metrics_total / len(clean_files)
    
    
    name_lst = ["PESQ_MOS", "CSIG", "CBAK", "COVL", "segSNR", "STOI"]
    
    for i in range(len(metrics_avg)):
    
        print(f"[INFO] {name_lst[i]}: {metrics_avg[i]:.4f}")
    
    
    print("\n\n[IFNO] Objective Scores of Thonburian Whisper Model")
    for key in thonburian_objectives:
        if key == "total_errors_words":
            print('{} = {:.4f}'.format("WER", thonburian_objectives[key] / thonburian_objectives['total_words']))
    
        elif key == "total_words_insertions":
            print('{} = {:.4f}'.format("IER_Word", thonburian_objectives[key] / thonburian_objectives['total_words']))
    
        elif key == "total_words_deletions":
            print('{} = {:.4f}'.format("DER_Word", thonburian_objectives[key] / thonburian_objectives['total_words']))
    
        elif key == "total_words_substitutions":
            print('{} = {:.4f}'.format("SER_Word", thonburian_objectives[key] / thonburian_objectives['total_words']))
    
        elif key == "total_errors_chars":
            print('{} = {:.4f}'.format("CER", thonburian_objectives[key] / thonburian_objectives['total_chars']))
    
        elif key == "total_chars_insertions":
            print('{} = {:.4f}'.format("IER_Char", thonburian_objectives[key] / thonburian_objectives['total_chars']))
    
        elif key == "total_chars_deletions":
            print('{} = {:.4f}'.format("DER_Char", thonburian_objectives[key] / thonburian_objectives['total_chars']))
    
        elif key == "total_chars_substitutions":
            print('{} = {:.4f}'.format("SER_Char", thonburian_objectives[key] / thonburian_objectives['total_chars']))
    
        elif key == "blue_1":
            print('{} = {:.4f}'.format(key, thonburian_objectives[key]/thonburian_objectives['count']))
    
        elif key == "blue_2":
            print('{} = {:.4f}'.format(key, thonburian_objectives[key]/thonburian_objectives['count']))
    
        elif key == "blue_4":
            print('{} = {:.4f}\n'.format(key, thonburian_objectives[key]/thonburian_objectives['count']))
    

    
# main function
if __name__ == '__main__':
    main()
    
    
