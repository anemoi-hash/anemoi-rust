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

/// Function state is set to 2 field elements or 64 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 2;
/// 1 element of the state is reserved for rate.
pub const RATE_WIDTH: usize = 1;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 1;

/// One element (32-bytes) is returned as digest.
pub const DIGEST_SIZE: usize = RATE_WIDTH;

/// The number of rounds is set to 19 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 19;

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

#[inline(always)]
/// Applies matrix-vector multiplication of the current
/// hash state with the Anemoi MDS matrix.
pub(crate) fn apply_mds(state: &mut [Felt; STATE_WIDTH]) {
    state[0] += mul_by_generator(&state[1]);
    state[1] += mul_by_generator(&state[0]);
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
                    0xaf32c9cf165eada2,
                    0xcd142aea4f8933a4,
                    0xadedcefcdb80a3f2,
                    0x2492eee7b78602d1,
                ])),
                Felt::new(BigInteger256([
                    0xed2e917c1a882a7b,
                    0x27555ffef6db88dc,
                    0xa57e3343a96e9edc,
                    0x308a8ff5968eb89c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x72e6052052fb2576,
                    0x7e374428e009cfad,
                    0x597542dc61002074,
                    0x0bc64e9c26176649,
                ])),
                Felt::new(BigInteger256([
                    0xc1683522e6331fe4,
                    0xc0199190995af6c9,
                    0x15ac8259fa6e5430,
                    0x24f8015a9a9106d3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6c016d433964cf86,
                    0x931cbf5ec579573d,
                    0xadb56010eb8891a1,
                    0x3c0ee074db8ebab0,
                ])),
                Felt::new(BigInteger256([
                    0x14947c8bf8189a2b,
                    0x3de68d3642af590f,
                    0x33074bdb2e1f32e6,
                    0x3af465ae18d8bc27,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6412b45207c577e6,
                    0x7a4c9ce0f401f986,
                    0x7a71b6a6c869b41a,
                    0x3b36908fae054acb,
                ])),
                Felt::new(BigInteger256([
                    0x1496cac521bbeb8e,
                    0xbf86e0f494da3ffa,
                    0x32b04029eea4b462,
                    0x31acd9bfdca2cc31,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x523ac48a8e0e05c1,
                    0xb4649dce2eda6556,
                    0xa6c3aab2ace026dc,
                    0x0262d1905c45a204,
                ])),
                Felt::new(BigInteger256([
                    0xf95b0524281495cd,
                    0x0189f8248c17e01b,
                    0x78ea24277b815a3d,
                    0x12469d360c41a2e4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x992fac04f34c7b20,
                    0x69ea30d7bc84d5d8,
                    0x80d9037b7344cc04,
                    0x0f4cd5e4b3110338,
                ])),
                Felt::new(BigInteger256([
                    0x313a02e918196965,
                    0xa7c28a6cd7c51f94,
                    0x80c786a969ca12f3,
                    0x03a23be04ad37cbb,
                ])),
            ],
        ];

        let output = [
            [
                Felt::new(BigInteger256([
                    0x123bd95299999999,
                    0xb83c0a9bfa40677b,
                    0xcccccccccccccccc,
                    0x0ccccccccccccccc,
                ])),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0xa942fcc6b6fc9019,
                    0xe30fee4001297b6b,
                    0xc81bbbb990652d5f,
                    0x151360a4ca1bade9,
                ])),
                Felt::new(BigInteger256([
                    0xc1f5f40ca5f0c922,
                    0x8ed096798f9a2565,
                    0x343bf2efffdc45e4,
                    0x136f8d515fc00212,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2daf6d68ef718427,
                    0x5061c937d5b3a764,
                    0xf8541bb755f5c730,
                    0x24e19c62db751b35,
                ])),
                Felt::new(BigInteger256([
                    0xffc6f91ea63fca57,
                    0x291c6ce7a87d6cc6,
                    0x0dde5b4ffa4ac0ea,
                    0x0d38cb1a91ae29c0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xeb95ce3a99999981,
                    0x819db2fb145092b5,
                    0xccccccccccccccc9,
                    0x0ccccccccccccccc,
                ])),
                Felt::new(BigInteger256([
                    0x311bac8400000004,
                    0x891a63f02652a376,
                    0x0000000000000000,
                    0x0000000000000000,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd33a705e05a2b501,
                    0xd2712374f0a94ed7,
                    0x685947bf80265288,
                    0x02f55f1c4483b157,
                ])),
                Felt::new(BigInteger256([
                    0x8d4b9f5d925218c6,
                    0xab82e5d35909713f,
                    0x99df4ced5436a059,
                    0x18f9479f97cecc77,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x56d459a2eec885c9,
                    0x800fff1ba5634f5e,
                    0xa9d619ce6174a00a,
                    0x16948d1bee84edc9,
                ])),
                Felt::new(BigInteger256([
                    0x39508316f7dc2903,
                    0xf5181fae2396063f,
                    0x764493e945645a4a,
                    0x2fcb2e0875254539,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xad59d6db2013cb29,
                    0x7d3ffdb8864f5d76,
                    0x6f0f792292a5da27,
                    0x002e3ad52a05f355,
                ])),
                Felt::new(BigInteger256([
                    0x40206688c617ff81,
                    0xd7e84f6aa98af8bc,
                    0x2b2cb6c5922777c0,
                    0x02f1936c3d1fe5f6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xda39c35ac8d7e335,
                    0x663f0bce1a55bbc7,
                    0x9633ef2e6c2165fc,
                    0x3d0234147d3015ff,
                ])),
                Felt::new(BigInteger256([
                    0xa24a0eec3834f53b,
                    0xc5784e751a48a557,
                    0x49912cd1195886ae,
                    0x122a29023bcfdf2a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1bd0c18129d6a6e9,
                    0x0e18df9c926d6ad0,
                    0x7764ddc92a8722ec,
                    0x17a4097a0928c447,
                ])),
                Felt::new(BigInteger256([
                    0xc427d902facec1a8,
                    0xfa944a4112663368,
                    0x148670bf732e1a83,
                    0x0c35060241f72690,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdc1eb5ce23e6ad24,
                    0x1f8fbfb03d0b4b70,
                    0xdc4e88da8a511dde,
                    0x345e553e92377642,
                ])),
                Felt::new(BigInteger256([
                    0xb265968cdcf050c2,
                    0x22734503bfe26972,
                    0xedf5906942f96c50,
                    0x15b372ccfece3041,
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
                    0xf3649672c4dd49c6,
                    0x869bfc1e41cb0082,
                    0x980f44810765c0eb,
                    0x228b774c79484a83,
                ])),
                Felt::new(BigInteger256([
                    0xcbfacca2abfb0f88,
                    0x50ba1a82b5bce92e,
                    0x764a2a27df58dbe8,
                    0x0ca35fdf4fc01011,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6cc606174f0c544f,
                    0x9326d9c83c3571c8,
                    0xa7f44d332eebd120,
                    0x03b9a2bf78be5af4,
                ])),
                Felt::new(BigInteger256([
                    0xf9cb7e913f7e3aa1,
                    0x1f87c23c1342b5e4,
                    0x8aa00ee2b98722b4,
                    0x0079da3042082d13,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2ef5f7a5aef20891,
                    0x86c88fb382964896,
                    0x3ef8dcbcb6874272,
                    0x0e449c509cba0c2a,
                ])),
                Felt::new(BigInteger256([
                    0xb584b19ee35ccfff,
                    0xe6083856c27653dd,
                    0xe6ca64144faa3396,
                    0x081c6d4d648df02f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3dfaf61934e83d59,
                    0x065e80cc9536d35c,
                    0xf863710dc64793fa,
                    0x20e36833a13c9320,
                ])),
                Felt::new(BigInteger256([
                    0x927a8954000adf21,
                    0x06637cf6e664c08d,
                    0x9ba24920b2b96867,
                    0x1e36c27103603c32,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf09b148a5aa6d195,
                    0xa2c0a47f0af263ea,
                    0x23644c6c27750b29,
                    0x1aa00852a6febd90,
                ])),
                Felt::new(BigInteger256([
                    0x4e16f65bf8882cec,
                    0xa5924aef470fecbd,
                    0x8ef9d1adfdf23719,
                    0x2b1881e2faf03fce,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe52a8bc20a094d52,
                    0x23d796fdc997611b,
                    0x7cf8401b0d8260e6,
                    0x05793f94fbd5065b,
                ])),
                Felt::new(BigInteger256([
                    0x35b85bb4801eec36,
                    0x337d951ec989acdb,
                    0x7d88342d55cb8392,
                    0x1c61ca0376eb0d08,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x65a0e008ffffffe9,
                    0xeba8415b23a4d418,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x99ed0724ffffff85,
                    0x88147ee76592dd8d,
                    0xffffffffffffffef,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x96bc8c8cffffffed,
                    0x74c2a54b49f7778e,
                    0xfffffffffffffffd,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x8f7765b8ffffff99,
                    0x3598729825300edc,
                    0xfffffffffffffff2,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::one(),
                Felt::new(BigInteger256([
                    0x96bc8c8cffffffed,
                    0x74c2a54b49f7778e,
                    0xfffffffffffffffd,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6303aa7f20c4976d,
                    0xf7f7e7afc4e6e58f,
                    0xe782174864220c74,
                    0x21bc56a908089ada,
                ])),
                Felt::new(BigInteger256([
                    0xa27f4adc4fd204a7,
                    0xe4046ef97b161340,
                    0xfbd49e91d4031a30,
                    0x3551112c77eb1657,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4dbf7eed8c837974,
                    0x30cda4f49c82ff41,
                    0x5d1497a0ce8f7ea5,
                    0x061ae5b0c2e73c56,
                ])),
                Felt::new(BigInteger256([
                    0x7e88f934fe0f99e5,
                    0x138bfb0321d1b22b,
                    0x5c070506c2549bee,
                    0x1f0056a4108c5ac3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xba8d6fc01fc2188c,
                    0x04f1a9654ee5ebea,
                    0xc0ecd12244da4465,
                    0x36d2bed3937fbd19,
                ])),
                Felt::new(BigInteger256([
                    0x292c33db82274ab7,
                    0x75a6236126a14bfd,
                    0xab6a79bfa7ed898f,
                    0x1a3a276f460ca1b0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x01d1ce7b351e98fc,
                    0xe1c2bfa702054465,
                    0x028edeb143e69dfc,
                    0x37f53468b21dc01e,
                ])),
                Felt::new(BigInteger256([
                    0x6a77e53809a3dc09,
                    0xe616d749ca2c7310,
                    0xa86ca297063a7e56,
                    0x3600c87c7df4fcc8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd23922f3354fb22e,
                    0x77c8503751840904,
                    0xee4564d21d301ea9,
                    0x321a91c18daffc98,
                ])),
                Felt::new(BigInteger256([
                    0x3818f8980316a7ce,
                    0x73617813b851765f,
                    0x3654c9c88fe2d068,
                    0x259d5aaabf602ecb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd93680068aa3ea5e,
                    0xe0be4e9fa61e6fa8,
                    0xf0a144fdba7bf2c0,
                    0x136231a64e6c4785,
                ])),
                Felt::new(BigInteger256([
                    0xe781f0b43552800b,
                    0x74ee8540fe8d3249,
                    0x30ae8d21fa374156,
                    0x3d4cc242ff0872a6,
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
