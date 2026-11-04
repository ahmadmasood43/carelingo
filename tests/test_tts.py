import numpy as np
import pytest

from carelingo.tts import VOICES, add_noise, pink_noise


def test_noise_is_mixed_at_the_requested_snr():
    rng = np.random.default_rng(0)
    signal = 0.3 * np.sin(np.linspace(0, 2000 * np.pi, 160_000))
    mixed = add_noise(signal, 20.0, rng)
    noise = mixed - signal
    snr = 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))
    assert snr == pytest.approx(20.0, abs=0.01)


def test_loud_mixes_are_scaled_instead_of_clipped():
    rng = np.random.default_rng(0)
    mixed = add_noise(np.full(16_000, 0.99), 0.0, rng)
    assert np.max(np.abs(mixed)) <= 1.0


def test_pink_noise_has_more_power_at_low_frequencies():
    power = np.abs(np.fft.rfft(pink_noise(2**16, np.random.default_rng(0)))) ** 2
    assert power[10:100].mean() > 10 * power[10_000:10_090].mean()


def test_every_language_has_a_native_voice():
    for voices in VOICES.values():
        assert any(voice.label == "native" for voice in voices)
