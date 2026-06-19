# import necessary packages
import torch



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


def reconstruct_waveform(spec, window, config, length, uncompress=False):

    if torch.is_complex(spec) and not uncompress:

        wav = torch.istft(spec, config.n_fft, config.hop, win_length=config.n_fft,
                        window=window, center=True, length=length)

    else:
        # get real and imaginary part
        real = spec[:, 0:1, :, :]
        imag = spec[:, 1:2, :, :]

        # reconstruct the audio from estimated real and imaginary part.
        spec_uncompress = power_uncompress(real, imag).squeeze(1)

        if not torch.is_complex(spec_uncompress):

            spec_uncompress = torch.view_as_complex(spec_uncompress)

        wav = torch.istft(spec_uncompress, config.n_fft, config.hop, win_length=config.n_fft,
                          window=window, center=True, length=length)

    return wav