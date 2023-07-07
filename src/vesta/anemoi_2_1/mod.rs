//! Implementation of the Anemoi permutation

use super::{sbox, BigInteger256, Felt};
use crate::{Anemoi, Jive, Sponge};
use ark_ff::{One, Zero};
/// Digest for Anemoi
mod digest;
/// Sponge for Anemoi
mod hasher;
/// Round constants for Anemoi
mod round_constants;

pub use digest::AnemoiDigest;

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

/// The number of rounds is set to 21 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 21;

// ANEMOI INSTANTIATION
// ================================================================================================

/// An Anemoi instantiation over Vesta basefield with 1 column and rate 1.
#[derive(Debug, Clone)]
pub struct AnemoiVesta_2_1;

impl<'a> Anemoi<'a, Felt> for AnemoiVesta_2_1 {
    const NUM_COLUMNS: usize = NUM_COLUMNS;
    const NUM_ROUNDS: usize = NUM_HASH_ROUNDS;

    const WIDTH: usize = STATE_WIDTH;
    const RATE: usize = RATE_WIDTH;
    const OUTPUT_SIZE: usize = DIGEST_SIZE;

    const ARK_C: &'a [Felt] = &round_constants::C;
    const ARK_D: &'a [Felt] = &round_constants::D;

    const GROUP_GENERATOR: u32 = sbox::BETA;

    const ALPHA: u32 = sbox::ALPHA;
    const INV_ALPHA: Felt = sbox::INV_ALPHA;
    const BETA: u32 = sbox::BETA;
    const DELTA: Felt = sbox::DELTA;

    fn exp_by_inv_alpha(x: Felt) -> Felt {
        sbox::exp_by_inv_alpha(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sbox() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
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
            AnemoiVesta_2_1::sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
