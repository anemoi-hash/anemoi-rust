//! Implementation of the Anemoi permutation

use super::{mul_by_generator, sbox, BigInteger256, Felt};
use crate::{Jive, Sponge};
use ark_ff::{Field, One, Zero};
use unroll::unroll_for_loops;

/// Digest for Anemoi
mod digest;
/// Sponge for Anemoi
mod hasher;

/// Round constants for Anemoi
mod round_constants;

pub use digest::AnemoiDigest;
pub use hasher::AnemoiHash;

// ANEMOI CONSTANTS
// ================================================================================================

/// Function state is set to 4 field elements or 128 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 4;
/// 3 elements of the state are reserved for rate.
pub const RATE_WIDTH: usize = 3;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 2;

/// One element (32-bytes) is returned as digest.
pub const DIGEST_SIZE: usize = 1;

/// The number of rounds is set to 14 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 14;

// HELPER FUNCTIONS
// ================================================================================================

/// Applies the Anemoi S-Box on the current
/// hash state elements.
#[inline(always)]
pub(crate) fn apply_sbox_layer(state: &mut [Felt; STATE_WIDTH]) {
    let mut x: [Felt; NUM_COLUMNS] = state[..NUM_COLUMNS].try_into().unwrap();
    let mut y: [Felt; NUM_COLUMNS] = state[NUM_COLUMNS..].try_into().unwrap();

    x.iter_mut().enumerate().for_each(|(i, t)| {
        let y2 = y[i].square();
        *t -= mul_by_generator(&y2);
    });

    let mut x_alpha_inv = x;
    x_alpha_inv
        .iter_mut()
        .for_each(|t| *t = sbox::exp_inv_alpha(t));

    y.iter_mut()
        .enumerate()
        .for_each(|(i, t)| *t -= x_alpha_inv[i]);

    x.iter_mut().enumerate().for_each(|(i, t)| {
        let y2 = y[i].square();
        *t += mul_by_generator(&y2) + sbox::DELTA;
    });

    state[..NUM_COLUMNS].copy_from_slice(&x);
    state[NUM_COLUMNS..].copy_from_slice(&y);
}

/// Applies matrix-vector multiplication of the current
/// hash state with the Anemoi MDS matrix.
#[inline(always)]
pub(crate) fn apply_linear_layer(state: &mut [Felt; STATE_WIDTH]) {
    state[0] += mul_by_generator(&state[1]);
    state[1] += mul_by_generator(&state[0]);

    state[3] += mul_by_generator(&state[2]);
    state[2] += mul_by_generator(&state[3]);
    state.swap(2, 3);

    // PHT layer
    state[2] += state[0];
    state[3] += state[1];

    state[0] += state[2];
    state[1] += state[3];
}

// ANEMOI PERMUTATION
// ================================================================================================

/// Applies an Anemoi permutation to the provided state
#[inline(always)]
#[unroll_for_loops]
pub(crate) fn apply_permutation(state: &mut [Felt; STATE_WIDTH]) {
    for i in 0..NUM_HASH_ROUNDS {
        apply_round(state, i);
    }

    apply_linear_layer(state)
}

/// Applies an Anemoi round to the provided state
#[inline(always)]
#[unroll_for_loops]
pub(crate) fn apply_round(state: &mut [Felt; STATE_WIDTH], step: usize) {
    // determine which round constants to use
    let c = &round_constants::C[step % NUM_HASH_ROUNDS];
    let d = &round_constants::D[step % NUM_HASH_ROUNDS];

    for i in 0..NUM_COLUMNS {
        state[i] += c[i];
        state[NUM_COLUMNS + i] += d[i];
    }

    apply_linear_layer(state);
    apply_sbox_layer(state);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sbox() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let mut input = [
            [Felt::zero(); 4],
            [Felt::one(); 4],
            [Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            [Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x3d800c82b5aa5e19,
                    0x2873c18451faed8a,
                    0xcfb65998526b1255,
                    0x30903f95eeb7cb29,
                ])),
                Felt::new(BigInteger256([
                    0x4c3f547e1b45f06d,
                    0x4434bfdc35cd2a7e,
                    0x2cec23911d5f09b1,
                    0x1e47bc39dbec80d6,
                ])),
                Felt::new(BigInteger256([
                    0x05b29901551cc8ac,
                    0x0b233ecda758a6cf,
                    0xa5fb456888de1061,
                    0x6989f6618c947c03,
                ])),
                Felt::new(BigInteger256([
                    0x5f30a926973d77b8,
                    0x58b9e04464714ec9,
                    0x6293a15bc67a2db2,
                    0x341ce33eebab7707,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xeb4cb4de1645b418,
                    0x5957542890e1d9f4,
                    0xeb1be24e2bcd2261,
                    0x1e9f83b8b26f534a,
                ])),
                Felt::new(BigInteger256([
                    0x53e3ac727664e2de,
                    0x7fc55b5ab9e39ae2,
                    0x28f650a20c8b37ed,
                    0x1a9930c5dcfed8bf,
                ])),
                Felt::new(BigInteger256([
                    0x6b8de7094420f58f,
                    0x1380d2431670bb3c,
                    0xb246b95b2645e282,
                    0x3bf17483e8673616,
                ])),
                Felt::new(BigInteger256([
                    0x0562472b2d5a94c0,
                    0xb57a227b3a10decf,
                    0x60a2f2227b1031ba,
                    0x253593281e2ebf56,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5bc39e099ce2390b,
                    0xf35c51e7ac84675c,
                    0xde5b7b78d171b0f4,
                    0x705f017f4de69193,
                ])),
                Felt::new(BigInteger256([
                    0xc60a598485c7da5e,
                    0xdb972cf05ecad741,
                    0x0ec32cdab5588a0a,
                    0x5db21faeff1c4ea5,
                ])),
                Felt::new(BigInteger256([
                    0xb48f6daeb282125d,
                    0xd66c2b8cf03578fd,
                    0xd3c811a2fb1af6e6,
                    0x2afb65bba7a45f2a,
                ])),
                Felt::new(BigInteger256([
                    0x11930798f52f2756,
                    0xbfc7961760cb118c,
                    0xb579f8ad3434a8e1,
                    0x521d2c1a498f560b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0ff35d71d6259c0f,
                    0x99b16bf9d35f80bb,
                    0xe67d27e275ff93dd,
                    0x49cb2e11ab65ecae,
                ])),
                Felt::new(BigInteger256([
                    0xf561238eea9ae2d5,
                    0x1467b9cf0110eeeb,
                    0x273e9de8fd873310,
                    0x0f0afb8dcee583b3,
                ])),
                Felt::new(BigInteger256([
                    0x1a92ecde353310bd,
                    0xcd654d14cdc6eafb,
                    0x6b4ca04c5b9d9e7f,
                    0x252e2928db32d186,
                ])),
                Felt::new(BigInteger256([
                    0x7f2548484d6ac6a5,
                    0xf8128cfc0c0b2ea6,
                    0x3d89cb7947434381,
                    0x4caf3c88595bbe02,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8d09cf76677403b7,
                    0x027ae25145361a5a,
                    0xf86955abeae4a5f4,
                    0x4712cfe8c517c328,
                ])),
                Felt::new(BigInteger256([
                    0xd4be943057dacaf1,
                    0x674e5276d7e1865e,
                    0xcacc08037a17aaad,
                    0x373cc1d452a5c765,
                ])),
                Felt::new(BigInteger256([
                    0x1f5a5a004d9a4aac,
                    0x043083c9cac67081,
                    0x45ba82827ce18d56,
                    0x4ced0e37ccd5434a,
                ])),
                Felt::new(BigInteger256([
                    0x50a706c0b4891aab,
                    0x87aebcab372f3d14,
                    0x127fb385025104d6,
                    0x19584e15865248fa,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9bb2415e56122242,
                    0xe62356b3961cd90a,
                    0xcc50d2e62ee1dd7b,
                    0x263f2e59e6b15b7d,
                ])),
                Felt::new(BigInteger256([
                    0xb804291fe0244ba4,
                    0xbe58dc3b960ba949,
                    0xd399b3450e9f2862,
                    0x09d85aa748743ff2,
                ])),
                Felt::new(BigInteger256([
                    0x155ab0c1788a610a,
                    0x64e59c1a11d49109,
                    0x975de11df8b09bee,
                    0x208098a9b2df008f,
                ])),
                Felt::new(BigInteger256([
                    0xa0480676095a5b11,
                    0x8a73c27278176b2b,
                    0x6f529bdfc9a40a71,
                    0x01ec7b738a67a465,
                ])),
            ],
        ];

        let output = [
            [
                Felt::new(BigInteger256([
                    0xdb6db6dadb6db6dc,
                    0xe6b5824adb6cc6da,
                    0xf8b356e005810db9,
                    0x66d0f1e660ec4796,
                ])),
                Felt::new(BigInteger256([
                    0xdb6db6dadb6db6dc,
                    0xe6b5824adb6cc6da,
                    0xf8b356e005810db9,
                    0x66d0f1e660ec4796,
                ])),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0xd99f4a8fe00e11bd,
                    0xe12a753d82b33b30,
                    0xba87bf86c47e6f0b,
                    0x58de312b5993b67f,
                ])),
                Felt::new(BigInteger256([
                    0xd99f4a8fe00e11bd,
                    0xe12a753d82b33b30,
                    0xba87bf86c47e6f0b,
                    0x58de312b5993b67f,
                ])),
                Felt::new(BigInteger256([
                    0xb4c1972e044567f4,
                    0x8e14aacbecb9fb59,
                    0x2c0f182612f81cd9,
                    0x34d6a5db27797543,
                ])),
                Felt::new(BigInteger256([
                    0xb4c1972e044567f4,
                    0x8e14aacbecb9fb59,
                    0x2c0f182612f81cd9,
                    0x34d6a5db27797543,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x34413b9512617cf9,
                    0x7e6bdaab3f8443c5,
                    0x752178862c9a6ab7,
                    0x3dde7b14d2951772,
                ])),
                Felt::new(BigInteger256([
                    0x34413b9512617cf9,
                    0x7e6bdaab3f8443c5,
                    0x752178862c9a6ab7,
                    0x3dde7b14d2951772,
                ])),
                Felt::new(BigInteger256([
                    0xbdf2fa0b931e7a04,
                    0x60f39c6eae7a9a3c,
                    0x34e97cf1bd62a467,
                    0x13b9e8a487cc4745,
                ])),
                Felt::new(BigInteger256([
                    0xbdf2fa0b931e7a04,
                    0x60f39c6eae7a9a3c,
                    0x34e97cf1bd62a467,
                    0x13b9e8a487cc4745,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdb6db6ecdb6db6ca,
                    0x035ffa14db8a4eec,
                    0x5ea2264f581fdd5a,
                    0x401b2e0d73d97883,
                ])),
                Felt::new(BigInteger256([
                    0xdb6db6ecdb6db6ca,
                    0x035ffa14db8a4eec,
                    0x5ea2264f581fdd5a,
                    0x401b2e0d73d97883,
                ])),
                Felt::new(BigInteger256([
                    0xfffffffd00000003,
                    0xfb38ec08fffb13fc,
                    0x99ad88181ce5880f,
                    0x5bc8f5f97cd877d8,
                ])),
                Felt::new(BigInteger256([
                    0xfffffffd00000003,
                    0xfb38ec08fffb13fc,
                    0x99ad88181ce5880f,
                    0x5bc8f5f97cd877d8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x14d9c6acfa768a3b,
                    0x98814b9b7de44345,
                    0xf887f587645b4c0e,
                    0x336d5d988eda4c9a,
                ])),
                Felt::new(BigInteger256([
                    0x4503956da1c041df,
                    0x9668bc719f92aace,
                    0x744a8e099d79a10c,
                    0x0ca52b989e9044cf,
                ])),
                Felt::new(BigInteger256([
                    0x47ead90d02ae4a00,
                    0x5452e0aead8079d5,
                    0xfc93f46ca19953cd,
                    0x334a1bce3144d7a9,
                ])),
                Felt::new(BigInteger256([
                    0x2d5f2046ede57cff,
                    0x6a20ddd6d3babab4,
                    0x2516328a622ddbf0,
                    0x1922b364af87de6c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0313fbc3455d4b6d,
                    0x3687f3b9b05e2c30,
                    0x8eef3e4a9aa936a3,
                    0x261074f6a391c626,
                ])),
                Felt::new(BigInteger256([
                    0x9dc680097f1de9c7,
                    0x49439c6a92b4bd2f,
                    0x88c22d3269736aba,
                    0x336e28a8c35107cf,
                ])),
                Felt::new(BigInteger256([
                    0xdf3d02c07438f5bb,
                    0x0e17d7991cdbcb1a,
                    0x78b53b19f43c491e,
                    0x6704d3bb96ce358b,
                ])),
                Felt::new(BigInteger256([
                    0x5a84e88a366f57f6,
                    0x8d5d3cca40a99b73,
                    0xbd8c9ce873086154,
                    0x4a5a45c9b9e51fc9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x16c5e80b94f60d5d,
                    0xcd7f1b24cb8cc8d4,
                    0x5847351762e87f59,
                    0x5769ce07b69f9d1e,
                ])),
                Felt::new(BigInteger256([
                    0x783876b380ea74fe,
                    0x6ccafb5519de3caf,
                    0xc4e87feab6548901,
                    0x144079a6cb96df85,
                ])),
                Felt::new(BigInteger256([
                    0x8c80c86983a0199c,
                    0x08249ec2f7a43109,
                    0x34a6647e54db2e51,
                    0x63bae0f9cc997aba,
                ])),
                Felt::new(BigInteger256([
                    0x7c8f9c8c71a83417,
                    0xcfebfba3ec22b353,
                    0x63a317b5b5a45ebf,
                    0x2a60c4a7fce4e5b0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x96efbee8e4894a16,
                    0x14207efbd3b6b61a,
                    0xd25890a70e02b3a0,
                    0x6bdd9302817c751e,
                ])),
                Felt::new(BigInteger256([
                    0x7789bb62a7160487,
                    0x4afa842d20af45ba,
                    0x09fd49753771d755,
                    0x0e44efd541b30c2e,
                ])),
                Felt::new(BigInteger256([
                    0x860f98063a560e8a,
                    0xc7e590d2a9a91b46,
                    0x07ebe01094526947,
                    0x4333399bd5c32ac7,
                ])),
                Felt::new(BigInteger256([
                    0x00af135028398f18,
                    0x14853c96ebccaf1b,
                    0xc86e541bbfee3c7a,
                    0x473345597a5b449b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0d4bd404753f6b96,
                    0x563f46f1800b636c,
                    0xb2a1cd6e7d10aaa0,
                    0x30a7f7b64aa112f2,
                ])),
                Felt::new(BigInteger256([
                    0x8b2d62b90adeb6eb,
                    0xa9b043315153acbf,
                    0x0a622d7474f9e15c,
                    0x4b7423fa5ab6aab5,
                ])),
                Felt::new(BigInteger256([
                    0x3c632cc64258a1d5,
                    0x1263ab333dcdba54,
                    0xc2a7a0d0cc42c9d2,
                    0x12f690cfbce8adc4,
                ])),
                Felt::new(BigInteger256([
                    0xa88a2163f847df8b,
                    0x14d25a1e46bf5a18,
                    0x6844e8837b4801a9,
                    0x0414393c9b3050b5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6bf41b0135417d1d,
                    0x70fec5f5acd2bed4,
                    0x3f6965571dbdee3a,
                    0x23ae011f2a718ccb,
                ])),
                Felt::new(BigInteger256([
                    0x74fe229a3040ac0d,
                    0x972ad8b45913805f,
                    0x6c5b32591f0c89f0,
                    0x650f90326ef83092,
                ])),
                Felt::new(BigInteger256([
                    0x218355fa379b3b5e,
                    0x074d1550b422e272,
                    0x3e50f6b04a87dfc5,
                    0x1c4aacaef1a99d1d,
                ])),
                Felt::new(BigInteger256([
                    0x52ee7ab3f0be73b7,
                    0x8a6d60fd8a8d3502,
                    0x593b7e106b057264,
                    0x3dddf2b51054ba69,
                ])),
            ],
        ];

        for i in input.iter_mut() {
            apply_sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
