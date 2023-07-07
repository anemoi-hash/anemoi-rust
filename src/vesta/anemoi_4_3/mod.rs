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

// ANEMOI INSTANTIATION
// ================================================================================================

/// An Anemoi instantiation over Vesta basefield with 2 columns and rate 3.
#[derive(Debug, Clone)]
pub struct AnemoiVesta_4_3;

impl<'a> Anemoi<'a, Felt> for AnemoiVesta_4_3 {
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
            [Felt::zero(); 4],
            [Felt::one(); 4],
            [Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            [Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x3898d6276e250a1a,
                    0x49ef67ce2abb8cd5,
                    0x67d3f7ac7aa535b1,
                    0x3be695163c5779ea,
                ])),
                Felt::new(BigInteger256([
                    0x07a443be9662fb4b,
                    0x781fadb683f44719,
                    0x2e93c0d94026aa4d,
                    0x0f9f9eb4964f5978,
                ])),
                Felt::new(BigInteger256([
                    0xe44e1c928faaad5b,
                    0x3e274fda062783e7,
                    0x6d4446c6a63f8126,
                    0x3f2595c15666d614,
                ])),
                Felt::new(BigInteger256([
                    0x1784c93382b00592,
                    0x860d15ad49ed3feb,
                    0x2c029d26007924bb,
                    0x1e24d3509bc5cebb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x40c257f79e66756f,
                    0x2740d9e2fad77116,
                    0x355155c0270ee015,
                    0x10666ec0fcb9f674,
                ])),
                Felt::new(BigInteger256([
                    0x862688a348b98cff,
                    0x5fccf768cc8241ec,
                    0x31be221dca2c996f,
                    0x3f417de5e31b44b6,
                ])),
                Felt::new(BigInteger256([
                    0x14015c2159cdc913,
                    0xbf61fba45f2ef946,
                    0x33135d0834e38671,
                    0x390c6566d5df6fad,
                ])),
                Felt::new(BigInteger256([
                    0x12224ed4d08c688d,
                    0xf228c8646f4936c1,
                    0x54e08e9821e0c1b0,
                    0x1595bbe195f2bf1c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4c9aa1d6b29ed549,
                    0xdb1b457283caa7ec,
                    0x64ebbc065f9b8df4,
                    0x0aa53d12795852d8,
                ])),
                Felt::new(BigInteger256([
                    0x4cf671ce385a018a,
                    0xacf8543356f9b541,
                    0xcb2b96fd651396fe,
                    0x2e26a10a41c5dacb,
                ])),
                Felt::new(BigInteger256([
                    0x645d9f95a9c56b1c,
                    0x022282bd008db7e1,
                    0xea2c55a1c02e4d93,
                    0x25eec9d245da0590,
                ])),
                Felt::new(BigInteger256([
                    0xb6cc96c5ce05aaef,
                    0x2118bbe41cab4473,
                    0x893c357f8887b564,
                    0x0f540ce8c8f4842a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x67018c5df6519203,
                    0xb9d4d7115150edfc,
                    0xfd53563d708296f4,
                    0x1b26c794741c0bfb,
                ])),
                Felt::new(BigInteger256([
                    0xfc62dfebfcb28f75,
                    0xaf3a0bb289ab9223,
                    0x82bb875fe1cae254,
                    0x384e091e28a6e58d,
                ])),
                Felt::new(BigInteger256([
                    0xd0fff7bf031a2ec9,
                    0xd5375bd9b2406182,
                    0xa0130e277ba41f2b,
                    0x0d019209e0f0fffe,
                ])),
                Felt::new(BigInteger256([
                    0xa445297a25803869,
                    0xb8b3a1ba5c67d015,
                    0x42760d37afb9b3f4,
                    0x34d4abee9150e48e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc49c6c9b2eab6c16,
                    0x2b14ddb75b3bbf69,
                    0x47933ee5da4bb5da,
                    0x38c2a8f5a624cde6,
                ])),
                Felt::new(BigInteger256([
                    0x3bff0cb9a49b6b75,
                    0xe32bbc6d129fb9b4,
                    0x210f9051874dcaf8,
                    0x114e00bf71772669,
                ])),
                Felt::new(BigInteger256([
                    0x08f372532d9f69c3,
                    0x4a425194827fe44c,
                    0x098d0028f362ce6a,
                    0x059ca7adb0f75bf6,
                ])),
                Felt::new(BigInteger256([
                    0xb9687d81a06577fe,
                    0x8bb1e9fd2c90903f,
                    0xe29c8a6906f72039,
                    0x1401c0cd3c01bfc9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x81c2c3c90a71cb0b,
                    0xf653c0858343a8e5,
                    0xd77e9e8ac1d45b02,
                    0x3069aa4adeecdfd2,
                ])),
                Felt::new(BigInteger256([
                    0xdfc550f74beb99d5,
                    0x5503dd57c62d70d4,
                    0x71c0e72f4626b9c7,
                    0x20dde63c8d6510ea,
                ])),
                Felt::new(BigInteger256([
                    0x60efa57c8292993f,
                    0x2e828c75d5d147ff,
                    0xbb5fc2e37fb6ed48,
                    0x3daecf10d4987081,
                ])),
                Felt::new(BigInteger256([
                    0x08b53593f3a2825a,
                    0x8227b9a271486308,
                    0x56e41268cc3a4b5a,
                    0x082feacb1be7237b,
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
                Felt::new(BigInteger256([
                    0x123bd95299999999,
                    0xb83c0a9bfa40677b,
                    0xcccccccccccccccc,
                    0x0ccccccccccccccc,
                ])),
                Felt::zero(),
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
                Felt::new(BigInteger256([
                    0x311bac8400000004,
                    0x891a63f02652a376,
                    0x0000000000000000,
                    0x0000000000000000,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfbcfdaf527dfeed2,
                    0x12c843702ae02f05,
                    0xe5677fac6c3e397c,
                    0x2f4922ff09d3c7d5,
                ])),
                Felt::new(BigInteger256([
                    0x1c332a154a53782f,
                    0x76e951a15bc540c6,
                    0xe3208d13688477c4,
                    0x01039cd670bd7a84,
                ])),
                Felt::new(BigInteger256([
                    0x665b3c8de3e8dc02,
                    0x10bfb169f067047e,
                    0x147ba05c7a24622a,
                    0x07bebec7c6acec01,
                ])),
                Felt::new(BigInteger256([
                    0x9b2b46fdb836640a,
                    0xe16d7db5694a8609,
                    0x26bd067ea71cd626,
                    0x27eb560e7d23b91e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfe823e540178f6a7,
                    0xe6df56615b3674c8,
                    0x2faaf4cee87de130,
                    0x07bbe3bafe3f518a,
                ])),
                Felt::new(BigInteger256([
                    0x739c338c54d2fe55,
                    0xe8b0cec326283d83,
                    0x7eecb4ead08ef622,
                    0x338db6b9cd75c264,
                ])),
                Felt::new(BigInteger256([
                    0x8babb83eb3ac2782,
                    0x01032f65f0fe68b0,
                    0x35e6c8e7de51894b,
                    0x04b49d8d3bf5e9a5,
                ])),
                Felt::new(BigInteger256([
                    0x123e14bbf16a6436,
                    0x0350e75a5b1fa5b7,
                    0x904262a2e247ada6,
                    0x0bba51838d4edf4c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x81e63de4667216aa,
                    0xee6b73405c02e799,
                    0xaafe47b932e8952e,
                    0x228a1ccc791aa519,
                ])),
                Felt::new(BigInteger256([
                    0x5f94679e3608dce2,
                    0x4bf6cbfecd8bfcb0,
                    0x64b8df309a21eef7,
                    0x2fac1e1a56940682,
                ])),
                Felt::new(BigInteger256([
                    0xdf04602f4d49e2df,
                    0x40b4c3ffd1461212,
                    0xeb6d8fa65f3dbc6b,
                    0x36c0df7c4fbd4b7c,
                ])),
                Felt::new(BigInteger256([
                    0x9af53e39ce820e12,
                    0xa8aec40f79cae035,
                    0x44e1804e7f9b4a09,
                    0x18c56bf644485615,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x04b00dd7113d1692,
                    0x7e02c3e3e4d4414e,
                    0x51d2ed025181aac7,
                    0x001e0e37b0bff1cb,
                ])),
                Felt::new(BigInteger256([
                    0x9e440667ca83b7c8,
                    0x674e17467df36f74,
                    0x1290ae479baa2741,
                    0x2be17968d5f0ac38,
                ])),
                Felt::new(BigInteger256([
                    0x86de9b30a6b7c042,
                    0x9a49e57c8ca0795e,
                    0xfd351d1a596615b0,
                    0x0cec52d8c421544a,
                ])),
                Felt::new(BigInteger256([
                    0xceb2af5bb19aa680,
                    0xb06c0ecf52c3d4fe,
                    0x49648df58c7e25c4,
                    0x19a1bff73ec23d04,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4a8dff05802409a7,
                    0x4f2f9311d37ade68,
                    0x95f54ed50b6e60da,
                    0x3842ffa543bb23a9,
                ])),
                Felt::new(BigInteger256([
                    0x0e129015807fe8db,
                    0xc5a9ae5ccca44e50,
                    0x915142c1b3f5b5c7,
                    0x0f764f19fb96ad90,
                ])),
                Felt::new(BigInteger256([
                    0x0423a6c005d2adb3,
                    0x573f72484253c322,
                    0x916e42da94cc487d,
                    0x0e71530d206eff2a,
                ])),
                Felt::new(BigInteger256([
                    0xe4d80d7bda046e62,
                    0x00b2ce60aa2da04e,
                    0xe363adb754a49307,
                    0x00acb620871a247a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x14971af392933ad5,
                    0x594a30d1bae47478,
                    0x4ed9b344b0fb7e55,
                    0x37f721f534104995,
                ])),
                Felt::new(BigInteger256([
                    0x89f30bd6ede1cffb,
                    0xb4f780c93f24381a,
                    0xbdc6c5cff8dfd85b,
                    0x1aa5ec6837db1d17,
                ])),
                Felt::new(BigInteger256([
                    0xcef78f2c31d44a62,
                    0x510b37dbc97cdcee,
                    0x9e3a8ab9cc8f1926,
                    0x006c4197590ff596,
                ])),
                Felt::new(BigInteger256([
                    0x635c0f809b89d105,
                    0xcb09bea77aa8a9f8,
                    0x5b5a6ceb8ef90767,
                    0x1866b1450e2d488e,
                ])),
            ],
        ];

        for i in input.iter_mut() {
            AnemoiVesta_4_3::sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
