//! Implementation of the Anemoi permutation

use super::{mul_by_generator, sbox, BigInteger256, Felt};
use crate::{Jive, Sponge};
use ark_ff::{Field, One, Zero};
use unroll::unroll_for_loops;

/// Digest for Anemoi
mod digest;
/// Sponge for Anemoi
mod hasher;
/// MDS matrix for Anemoi
mod mds;
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

/// The number of rounds is set to 12 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 12;

// HELPER FUNCTIONS
// ================================================================================================

/// Applies the Anemoi S-Box on the current
/// hash state elements.
#[inline(always)]
pub(crate) fn apply_sbox(state: &mut [Felt; STATE_WIDTH]) {
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
pub(crate) fn apply_mds(state: &mut [Felt; STATE_WIDTH]) {
    state[0] += mul_by_generator(&state[1]);
    state[1] += mul_by_generator(&state[0]);

    state[3] += mul_by_generator(&state[2]);
    state[2] += mul_by_generator(&state[3]);
    state.swap(2, 3);
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

    apply_mds(state)
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

    apply_mds(state);
    apply_sbox(state);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_naive_mds(state: &mut [Felt; STATE_WIDTH]) {
        let x: [Felt; NUM_COLUMNS] = state[..NUM_COLUMNS].try_into().unwrap();
        let mut y: [Felt; NUM_COLUMNS] = [Felt::zero(); NUM_COLUMNS];
        y[0..NUM_COLUMNS - 1].copy_from_slice(&state[NUM_COLUMNS + 1..]);
        y[NUM_COLUMNS - 1] = state[NUM_COLUMNS];

        let mut result = [Felt::zero(); STATE_WIDTH];
        for (i, r) in result.iter_mut().enumerate().take(NUM_COLUMNS) {
            for (j, s) in x.into_iter().enumerate().take(NUM_COLUMNS) {
                *r += s * mds::MDS[i * NUM_COLUMNS + j];
            }
        }
        for (i, r) in result.iter_mut().enumerate().skip(NUM_COLUMNS) {
            for (j, s) in y.into_iter().enumerate() {
                *r += s * mds::MDS[(i - NUM_COLUMNS) * NUM_COLUMNS + j];
            }
        }

        state.copy_from_slice(&result);
    }

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
            apply_sbox(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }

    #[test]
    fn test_mds() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let mut input = [
            [Felt::zero(); 4],
            [Felt::one(); 4],
            [Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            [Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x86ac2b4969239685,
                    0xdcbbc64cff3aa846,
                    0xdff01a146089467b,
                    0x2e4f708e205707f3,
                ])),
                Felt::new(BigInteger256([
                    0xde0fcbfa2ff001be,
                    0xde2168b9fc0d1b1f,
                    0x29bfc1ae6f175bc3,
                    0x0eafd62586d97561,
                ])),
                Felt::new(BigInteger256([
                    0x8e801af1bc15620d,
                    0xfcb5b9ae3795c24d,
                    0xcf1207146008a11a,
                    0x43d2bd3c0ae5eda3,
                ])),
                Felt::new(BigInteger256([
                    0xb73450a6525fe9ff,
                    0xeff7381cfc9a8564,
                    0x79a89b08df64b2a7,
                    0x5a32bf1bf21e17ac,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x686be5457ee7fb62,
                    0x89cf5a7a034d2ab5,
                    0x445efc3749bf4c53,
                    0x6bce8ae1e0b030a6,
                ])),
                Felt::new(BigInteger256([
                    0x66e4824dd19ba533,
                    0x24ae439ce04418a0,
                    0x83aa7bc9addb1a42,
                    0x5ceb35013b0c9e94,
                ])),
                Felt::new(BigInteger256([
                    0xe7bec5909c5b1dad,
                    0x39c9add27c77aaeb,
                    0x422eea3d5cdcd730,
                    0x452e6b6b47af9f95,
                ])),
                Felt::new(BigInteger256([
                    0xecbdf0471ae65614,
                    0x28d47a07a9be572f,
                    0x40060c33286e7ff5,
                    0x0a793f12b62f68eb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5eb61b98bf52d264,
                    0xd5d483927552e55b,
                    0x6eb22f8c85b747ce,
                    0x326c8f4d36cae717,
                ])),
                Felt::new(BigInteger256([
                    0xe847998917f6f8c6,
                    0xadd65a71cbd86365,
                    0x52d73284ee201f10,
                    0x489dca2476a05e43,
                ])),
                Felt::new(BigInteger256([
                    0x3e528e6c7975470f,
                    0xb32a1c56b9d81920,
                    0xc60e7f0d5bca7d3c,
                    0x07eedf09805507f5,
                ])),
                Felt::new(BigInteger256([
                    0xbe6cfb24577887b0,
                    0x40fea57dfe9b5ff8,
                    0xe2d6b3b1449f9d6f,
                    0x4651d540dcb7be88,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd06505bd08994954,
                    0x110d7d9d49ff89f5,
                    0x0ffbc734ee8135ba,
                    0x6bb979f678e2f3e1,
                ])),
                Felt::new(BigInteger256([
                    0xe9a7ba84526fe1d9,
                    0x186af5ac3ef1d20c,
                    0x55bfed7c2c8c3c1d,
                    0x59234f91dc4421f2,
                ])),
                Felt::new(BigInteger256([
                    0xa6b17bb025f1d7d3,
                    0x5c1e2ec4e517e8f2,
                    0x5a9258eea4ee0af2,
                    0x6c4eef0aedd60649,
                ])),
                Felt::new(BigInteger256([
                    0xbf7ebee0473308c8,
                    0xec7d39bd2285a553,
                    0x29043b642b6b17ca,
                    0x1d29911e015d5126,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x68e9739bf6a08377,
                    0x545f557f977a6c93,
                    0xe41e77fbe35c01e8,
                    0x6e3bf0d6c5e226f0,
                ])),
                Felt::new(BigInteger256([
                    0x052641388645de02,
                    0x020e79334dcc0142,
                    0x54d42963cd5e8712,
                    0x6e46fe5e14aa1568,
                ])),
                Felt::new(BigInteger256([
                    0xe36e4df35122dd7e,
                    0x97c7a475cdfa9122,
                    0x0d10e231c73a7119,
                    0x713d5033f7b8e68a,
                ])),
                Felt::new(BigInteger256([
                    0xbcea985a3f396394,
                    0xc16358a9915523b1,
                    0x6d068c3e66b902f2,
                    0x66aeaf72c184204a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x313b6e69c50af819,
                    0x676c3998d71f6ce7,
                    0xca10537097f10b31,
                    0x59942a586ef2faf4,
                ])),
                Felt::new(BigInteger256([
                    0x2024a604173dc06e,
                    0x135a2c7342638d26,
                    0xe922d941de38b636,
                    0x6833e6a8b1d92878,
                ])),
                Felt::new(BigInteger256([
                    0x2922bd4af28a0d7e,
                    0xd04afd57011dd81a,
                    0x7ef65d060f1d64c2,
                    0x31aa12cbf4c68df7,
                ])),
                Felt::new(BigInteger256([
                    0x48baaa098a935415,
                    0x42b6fdb174763ef1,
                    0x919976cd067a33a7,
                    0x6de1904529d49e30,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(); 4],
            [
                Felt::new(BigInteger256([
                    0x00000010ffffffef,
                    0x70681bcd001be411,
                    0x9928a7775c40a7a5,
                    0x4d37e37a3c8aae34,
                ])),
                Felt::new(BigInteger256([
                    0x0000007cffffff83,
                    0x1c66ea8900cd147d,
                    0xfcc184134bf98566,
                    0x64f54c64ae19d3be,
                ])),
                Felt::new(BigInteger256([
                    0x00000010ffffffef,
                    0x70681bcd001be411,
                    0x9928a7775c40a7a5,
                    0x4d37e37a3c8aae34,
                ])),
                Felt::new(BigInteger256([
                    0x0000007cffffff83,
                    0x1c66ea8900cd147d,
                    0xfcc184134bf98566,
                    0x64f54c64ae19d3be,
                ])),
            ],
            [
                Felt::zero(),
                Felt::zero(),
                Felt::new(BigInteger256([
                    0x00000010ffffffef,
                    0x70681bcd001be411,
                    0x9928a7775c40a7a5,
                    0x4d37e37a3c8aae34,
                ])),
                Felt::new(BigInteger256([
                    0x0000007cffffff83,
                    0x1c66ea8900cd147d,
                    0xfcc184134bf98566,
                    0x64f54c64ae19d3be,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x00000010ffffffef,
                    0x70681bcd001be411,
                    0x9928a7775c40a7a5,
                    0x4d37e37a3c8aae34,
                ])),
                Felt::new(BigInteger256([
                    0x0000007cffffff83,
                    0x1c66ea8900cd147d,
                    0xfcc184134bf98566,
                    0x64f54c64ae19d3be,
                ])),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0x991abf21b8b3a2b6,
                    0x9be7ff5fe3980a26,
                    0xd0f48dd1608af0d1,
                    0x2130a441a6abc053,
                ])),
                Felt::new(BigInteger256([
                    0x0dcb05e83cd974b6,
                    0x79fe1c533538aa30,
                    0x79fbf257ffa04174,
                    0x0f29054ac250bd1b,
                ])),
                Felt::new(BigInteger256([
                    0x9cb50d4676f59856,
                    0x89f8bbd481b96587,
                    0x563f6c775919ba4f,
                    0x653f4e7397f1a206,
                ])),
                Felt::new(BigInteger256([
                    0xd77377e4fccc8c61,
                    0xcc11046bc3b16108,
                    0xf772ee2795f1a927,
                    0x50fbf67238d06c1e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x38ab756c3a297fc1,
                    0x94215bb22533af1e,
                    0xaaad4e8ad0f1f402,
                    0x3eab11f784579704,
                ])),
                Felt::new(BigInteger256([
                    0xf394b84768be2376,
                    0xe2a1356fe4b47277,
                    0x618041753df1663e,
                    0x43e2157932fbca94,
                ])),
                Felt::new(BigInteger256([
                    0x42f5573f616425cb,
                    0x6e61aabd110a93a7,
                    0x426713c08bf10231,
                    0x1f0791b50586d0df,
                ])),
                Felt::new(BigInteger256([
                    0xbc74284e46182638,
                    0x96fa10f7f3c4fc80,
                    0xac8cc4711d30367f,
                    0x368818b81b245b1f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb8ab4e5c67139fca,
                    0x47ba6ca308442d28,
                    0xe5ad310ee210c12e,
                    0x5f0678ffceb785cd,
                ])),
                Felt::new(BigInteger256([
                    0xf4f6be15e9805746,
                    0xad7d7ad505bf7788,
                    0x673879bce2ca5734,
                    0x2a392d3023f41733,
                ])),
                Felt::new(BigInteger256([
                    0x72aee01ca9ad7918,
                    0xd367c7da1385b3db,
                    0x1a025506bd873212,
                    0x09ec4730356d78f9,
                ])),
                Felt::new(BigInteger256([
                    0x611aaf351d3396b7,
                    0x7b00934d42800420,
                    0x7c1ed23c8a7cdbc0,
                    0x4d64d15af65356c5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x33fb1f6149a8763d,
                    0xc5885d4102a62056,
                    0x34e03569ec8bca65,
                    0x241ebb00850ef1d0,
                ])),
                Felt::new(BigInteger256([
                    0x5585962f560b1d82,
                    0xd7aa3a6d517ffc6a,
                    0x616db351911b14da,
                    0x6e1f1def2c71c413,
                ])),
                Felt::new(BigInteger256([
                    0x4e5920b750cfef87,
                    0x7adea90d6636dbfc,
                    0x6fa999ba7422544b,
                    0x5bc02e7788868d76,
                ])),
                Felt::new(BigInteger256([
                    0xcb2160b95ba1647e,
                    0xc1c2f610b0a1c4de,
                    0x34da7cd7981348e2,
                    0x36fe485cafd2f4d5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8cf53c2ea289957e,
                    0x189529d1b819f168,
                    0xcf56b17e3d84cb41,
                    0x46ad5223333a4fd1,
                ])),
                Felt::new(BigInteger256([
                    0xdfdae683f808f46f,
                    0x0b6e69e05689cf22,
                    0x0011cbaf4bd6ddbf,
                    0x195ff8b4ab2ed1bc,
                ])),
                Felt::new(BigInteger256([
                    0xf4eeba08772d71ff,
                    0x9da95bcd333a97ac,
                    0x61e7d36295e53280,
                    0x53dc4d9864440117,
                ])),
                Felt::new(BigInteger256([
                    0x95f564349360fb71,
                    0xf0f74f00349e8ee3,
                    0x870c99b3a6b3c27d,
                    0x04b1836bbbe3fe7c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x123bf88d67bb3b14,
                    0xa4b3f4aaa7e3c4f9,
                    0xc36f5c0568111e86,
                    0x077fe5af2894a949,
                ])),
                Felt::new(BigInteger256([
                    0x9fc871e2ed5c5df9,
                    0x4088391ad99f93f6,
                    0x0df4855fad0eb3df,
                    0x28c5871fa44c4c35,
                ])),
                Felt::new(BigInteger256([
                    0x68add7192c59b284,
                    0xf98aff097c4c13ab,
                    0x70a879df53626cea,
                    0x6dbf1ddf5e6a081c,
                ])),
                Felt::new(BigInteger256([
                    0x05e39f0228fdef13,
                    0x58e87a84673dddd1,
                    0x2cfcc9e91361770a,
                    0x066450a1665e59c5,
                ])),
            ],
        ];

        for i in input.iter_mut() {
            apply_mds(i);
        }
        for i in input2.iter_mut() {
            apply_naive_mds(i);
        }

        for (index, (&i_1, i_2)) in input.iter().zip(input2).enumerate() {
            assert_eq!(output[index], i_1);
            assert_eq!(output[index], i_2);
        }
    }
}
