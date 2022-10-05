use super::{sbox, BigInteger256, Felt};
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

/// Function state is set to 2 field elements or 64 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 2;
/// 1 element of the state is reserved for rate.
pub const RATE_WIDTH: usize = 1;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 1;

/// Two elements (64-bytes) is returned as digest.
// This is necessary to ensure 256 bits security.
pub const DIGEST_SIZE: usize = 2;

/// The number of rounds is set to 35 to provide 256-bit security level.
pub const NUM_HASH_ROUNDS: usize = 35;

// HELPER FUNCTIONS
// ================================================================================================

#[inline(always)]
/// Applies exponentiation of the current hash
/// state elements with the Anemoi S-Box.
pub(crate) fn apply_sbox(state: &mut [Felt; STATE_WIDTH]) {
    let mut x: [Felt; NUM_COLUMNS] = state[..NUM_COLUMNS].try_into().unwrap();
    let mut y: [Felt; NUM_COLUMNS] = state[NUM_COLUMNS..].try_into().unwrap();

    x.iter_mut().enumerate().for_each(|(i, t)| {
        let y2 = y[i].square();
        let beta_y2 = y2 + y2.double().double();
        *t -= beta_y2;
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
        let beta_y2 = y2 + y2.double().double();
        *t += beta_y2 + sbox::DELTA;
    });

    state[..NUM_COLUMNS].copy_from_slice(&x);
    state[NUM_COLUMNS..].copy_from_slice(&y);
}

#[inline(always)]
/// Applies matrix-vector multiplication of the current
/// hash state with the Anemoi MDS matrix.
pub(crate) fn apply_mds(state: &mut [Felt; STATE_WIDTH]) {
    let xy: [Felt; NUM_COLUMNS + 1] = [state[0], state[1]];

    let tmp = xy[1] * mds::MDS[1];
    state[0] = xy[0] + tmp;
    state[1] = (tmp + xy[0]) * mds::MDS[1] + xy[1];
}

// ANEMOI PERMUTATION
// ================================================================================================

/// Applies Anemoi permutation to the provided state.
#[inline(always)]
#[unroll_for_loops]
pub(crate) fn apply_permutation(state: &mut [Felt; STATE_WIDTH]) {
    for i in 0..NUM_HASH_ROUNDS {
        apply_round(state, i);
    }

    apply_mds(state)
}

/// Anemoi round function;
/// implementation based on algorithm 3 of <https://eprint.iacr.org/2020/1143.pdf>
#[inline(always)]
pub(crate) fn apply_round(state: &mut [Felt; STATE_WIDTH], step: usize) {
    state[0] += round_constants::C[step % NUM_HASH_ROUNDS];
    state[1] += round_constants::D[step % NUM_HASH_ROUNDS];

    apply_mds(state);
    apply_sbox(state);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_naive_mds(state: &mut [Felt; STATE_WIDTH]) {
        let mut result = [Felt::zero(); STATE_WIDTH];
        for (i, r) in result.iter_mut().enumerate().take(STATE_WIDTH) {
            for (j, s) in state.iter().enumerate().take(STATE_WIDTH) {
                *r += *s * mds::MDS[i * STATE_WIDTH + j];
            }
        }

        state.copy_from_slice(&result);
    }

    #[test]
    fn test_sbox() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let mut input = [
            [Felt::zero(), Felt::zero()],
            [Felt::one(), Felt::one()],
            [Felt::zero(), Felt::one()],
            [Felt::one(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x71c447c50b9a0026,
                    0xb4da83ee924c230e,
                    0x5a654fbf7acd0489,
                    0x35ceaf9fa254b921,
                ])),
                Felt::new(BigInteger256([
                    0x7c65ec076e637ab6,
                    0x37cc67f8636d539d,
                    0x21da745d60a03e79,
                    0x3999e31db2d6eeb6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3ee4616255f806eb,
                    0x438013243395cb79,
                    0x887fab81c146eae2,
                    0x1a73c2db7bc28a44,
                ])),
                Felt::new(BigInteger256([
                    0xad1e968215714e0d,
                    0x330d5b431a221ea0,
                    0x60417a3384e7a58e,
                    0x276fa7dc9ba23948,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa40d17b20bc49f92,
                    0x0193c40cd8ef25d9,
                    0x313441901be374e9,
                    0x3a872c090e7508a4,
                ])),
                Felt::new(BigInteger256([
                    0x2ab3317f92692e61,
                    0xf39bd2d2d62c5fa0,
                    0x7d892007dd7226dc,
                    0x3954a190d4432807,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc6ce14387863c444,
                    0x992ba3f15fb03ec8,
                    0xfdc8af1801814867,
                    0x06dbf4e431958d93,
                ])),
                Felt::new(BigInteger256([
                    0xbdcb680936b86d8c,
                    0x4657687d6020248d,
                    0x2ec1cd279ae49bc9,
                    0x34ae283334ede7f7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x371e51bb19a5bb03,
                    0x0799a948f22963df,
                    0xa39291acd6fdeb56,
                    0x182dc3c3c1074c0f,
                ])),
                Felt::new(BigInteger256([
                    0x6e8e16890115389b,
                    0xc7aff8c3d7ed0e00,
                    0x715ac29b01ff6676,
                    0x3c1cc3e22b6eca6c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd6089b02a5b6c9bb,
                    0xbbe3c86f95c78408,
                    0x2c3054059c87a06e,
                    0x2bb9a6b1f5ff2283,
                ])),
                Felt::new(BigInteger256([
                    0xa145ec9842e57217,
                    0xa9000e1d8aa222f2,
                    0x331b74ee8dd0e5f2,
                    0x0e7695043f73f059,
                ])),
            ],
        ];

        let output = [
            [
                Felt::new(BigInteger256([
                    0x0a7e7c3e99999999,
                    0xb83c0a9bfa6b6a89,
                    0xcccccccccccccccc,
                    0x0ccccccccccccccc,
                ])),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0x3785babe3451890e,
                    0xbad6a04b96e7e335,
                    0xdaecef0c6271ae56,
                    0x309b6fa4569ce2e5,
                ])),
                Felt::new(BigInteger256([
                    0x376eae911f9c7044,
                    0xf15d13a83fed92e2,
                    0x8a6f1eb3ded38974,
                    0x0a63a5046381f69e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe89d156895f33598,
                    0x22cc8f90197cdefa,
                    0x59d1af10ece372fe,
                    0x3c38ccf4eeef1fb8,
                ])),
                Felt::new(BigInteger256([
                    0x1fae7c1b952b2763,
                    0x6ff92987f72625f6,
                    0x05ce29c9b82e2059,
                    0x02ee5a311dcf8a77,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xae41e60699999981,
                    0x819db2fb1b340ff2,
                    0xccccccccccccccc9,
                    0x0ccccccccccccccc,
                ])),
                Felt::new(BigInteger256([
                    0x64b4c3b400000004,
                    0x891a63f02533e46e,
                    0x0000000000000000,
                    0x0000000000000000,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc41b08c9b25f26bc,
                    0x20203114a1f40102,
                    0x2accf016319287d8,
                    0x0037aeed01f801ee,
                ])),
                Felt::new(BigInteger256([
                    0x215421da09b22459,
                    0x8e7f3497f0ae5261,
                    0x885fa727affea6ab,
                    0x2572078f7ee945d3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x97fed3d25b3bcd64,
                    0xdec8ab1dac464faf,
                    0x2aaa9fdc01b48822,
                    0x10096dc9cdf356db,
                ])),
                Felt::new(BigInteger256([
                    0xe9e5cd32f0ace3c6,
                    0xdb91309fee7fc5e3,
                    0x74853694d09d8a4a,
                    0x0cd42ba79d4a79a7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x73e1a206b96e8399,
                    0x4e29df4e95fedbeb,
                    0xbf5473277d49c724,
                    0x2e1bb095b0dca7b0,
                ])),
                Felt::new(BigInteger256([
                    0xc55648041f536604,
                    0x8d0b308972fc4593,
                    0xc74daa59f3b75ba2,
                    0x1164d5e5362b1a4d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb18fc951b1f7e576,
                    0xe7d3f37d26a54490,
                    0x27d7a4236266c002,
                    0x07f668daa13dd686,
                ])),
                Felt::new(BigInteger256([
                    0xf29d7fb0a56f8dfa,
                    0x3aeaad1a8411ce98,
                    0x5a6ed122ae284f8e,
                    0x0f85a5cbc13d4627,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf5e36693cbd19d2b,
                    0xb69875d55ec48f35,
                    0xa71520ad4c55100b,
                    0x0b8365922e81c6e0,
                ])),
                Felt::new(BigInteger256([
                    0xb734aaf5e1865e3b,
                    0x078d8dada4792dae,
                    0xebf32f929ca14bca,
                    0x002100e669a5d630,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x26886fe56c820332,
                    0x3e2a589e35bd7f5d,
                    0xf1f4f5962eafc3c0,
                    0x31c597f486a60225,
                ])),
                Felt::new(BigInteger256([
                    0xb772e5273f5d48e6,
                    0x27fe09e7f0ca9dba,
                    0xd6d41cd85b200b6f,
                    0x3a62f520e56cefec,
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
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let mut input = [
            [Felt::zero(), Felt::zero()],
            [Felt::one(), Felt::one()],
            [Felt::zero(), Felt::one()],
            [Felt::one(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x6f8d73bdd58a318b,
                    0x9d6f60531d96fe39,
                    0x9e63ea457b0f4843,
                    0x3aa49a5494811197,
                ])),
                Felt::new(BigInteger256([
                    0xaef57d16dbcb0f74,
                    0x3b350b8dff5e7d88,
                    0xa5f6fd3b96e8ea35,
                    0x1e4a842aacabd9e9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe157f0da0658b276,
                    0x2750bedfecffc02b,
                    0x2ddb10f7d7996f96,
                    0x3648fd71695c335a,
                ])),
                Felt::new(BigInteger256([
                    0xbf00522b390f9d69,
                    0x9c17ff7966712660,
                    0xe2909219a6f94243,
                    0x07ba47cc30263411,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb0d1fb123de45c92,
                    0x53c12ef41815feeb,
                    0x697eebd468057444,
                    0x1301adb4bb539d41,
                ])),
                Felt::new(BigInteger256([
                    0x2d74b7bc54eddbc6,
                    0xafb6396e4ab2ea4f,
                    0x9d8ec26acbabc775,
                    0x37066d9e3f83635f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6cf6d18a495c2c50,
                    0xd54372d6731aadc5,
                    0xcbd7be25d3f3510f,
                    0x0b84d349724076a8,
                ])),
                Felt::new(BigInteger256([
                    0xe959cedf43704f2f,
                    0xfcfe0f033a5fae1b,
                    0xdb7c1783a976427a,
                    0x047051210d7a9809,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x47e05e83cf578b11,
                    0x1de347a03ffd277c,
                    0x28c01df1d6dd7add,
                    0x1820ce722098dc18,
                ])),
                Felt::new(BigInteger256([
                    0xc502c622e877930a,
                    0xa5603a26650be19c,
                    0xdd8d3cf568b5cb60,
                    0x32d24aee341132f4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x25781e7b4539db3b,
                    0xa519b38309412fed,
                    0x376403f9d1651827,
                    0x270cb25bd01f43ad,
                ])),
                Felt::new(BigInteger256([
                    0xb9c8e73f00f4121f,
                    0xed58378520a74259,
                    0x8317da90cdedf7c3,
                    0x02f244c8183fb0b3,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x3cf09ab4ffffffe9,
                    0xeba8415b2a159e85,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x67497e20ffffff85,
                    0x88147ee788044fbd,
                    0xffffffffffffffef,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa1a55e68ffffffed,
                    0x74c2a54b4f4982f3,
                    0xfffffffffffffffd,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x5ed150a4ffffff99,
                    0x359872984207c5e5,
                    0xfffffffffffffff2,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::one(),
                Felt::new(BigInteger256([
                    0xa1a55e68ffffffed,
                    0x74c2a54b4f4982f3,
                    0xfffffffffffffffd,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0ed1526920817ecc,
                    0x5ea4cf24fe888692,
                    0xdc36dc6f6d9bdb4d,
                    0x12192f29f3dc5327,
                ])),
                Felt::new(BigInteger256([
                    0x5fdee8377e52896f,
                    0xf2267e4aeebc2547,
                    0xf3094b68baf432b7,
                    0x38c86ffc6ff979b0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x032c5ac523a6c582,
                    0x11822342e3e886f4,
                    0x9aadeb781a77bae8,
                    0x1cec646e5a1b37b3,
                ])),
                Felt::new(BigInteger256([
                    0x9c83b62aeb5178f1,
                    0xaf157dcfc761d6ed,
                    0xe7f62b722b4fe8cb,
                    0x18583df3f2ae4a93,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2f64ce0be689a76c,
                    0x3935ea2b6860ae09,
                    0x7d48b7ea62605990,
                    0x2621d1cbf8e48e1f,
                ])),
                Felt::new(BigInteger256([
                    0x4ee52b30d59e20df,
                    0x66f0015338af652a,
                    0x0ffa59feb78d8746,
                    0x35af869a1bfa29fd,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfbb7dbe69a8db83b,
                    0xc639bde696f91450,
                    0x154433b823429d76,
                    0x21b668eeb5a56eda,
                ])),
                Felt::new(BigInteger256([
                    0xa196b8864834e854,
                    0x9791928c1aa32179,
                    0x45d11a1c59c355cc,
                    0x2d005dca99b5c24c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbc39797e59ad6a3f,
                    0xcfaa06701404ab1d,
                    0x7c824ebce26a73bf,
                    0x163c451924eedae0,
                ])),
                Felt::new(BigInteger256([
                    0x3fc7c3c0a8daa643,
                    0x6f25285eb68946fa,
                    0x4c18c6a5d4ca0e1f,
                    0x21ffa46becbb7957,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc664a2b649fe35d6,
                    0x47d2c91cac857bad,
                    0xc6db48cdd70aeefb,
                    0x35c80a44495db72e,
                ])),
                Felt::new(BigInteger256([
                    0x350b511a72eb1f49,
                    0xcb5bc1245a0ec850,
                    0x656046960124a2ab,
                    0x0fda781d8714449d,
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
