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

/// An Anemoi instantiation over Pallas basefield with 2 columns and rate 3.
#[derive(Debug, Clone)]
pub struct AnemoiPallas_4_3;

impl<'a> Anemoi<'a, Felt> for AnemoiPallas_4_3 {
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
                    0x3263957d45a582f5,
                    0x9811d8edb5d76596,
                    0xe7c4ff218fb2e0ad,
                    0x31fee62e20ee8cc8,
                ])),
                Felt::new(BigInteger256([
                    0xa482f8f57e3f9a09,
                    0x4330b1c95b1a9f15,
                    0xd6fa3dbfb5b30e23,
                    0x353a1859622d16ed,
                ])),
                Felt::new(BigInteger256([
                    0xb4ed60a5581a4fd5,
                    0xf4bdac3f53b4ef15,
                    0xa357254c9fa2dbf3,
                    0x3f9c122b98176962,
                ])),
                Felt::new(BigInteger256([
                    0x3eeeb90c8f23f687,
                    0xc3e67899e2938f94,
                    0x76896c2ff4a43ae4,
                    0x23072ac2a58269e2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x31035a0548abd251,
                    0xa25ec7d59322e797,
                    0x23b02f37a4d9d38b,
                    0x299fe02d86631e71,
                ])),
                Felt::new(BigInteger256([
                    0xfb9bf25282ee0c54,
                    0x35518a9a50e0a60e,
                    0xad90c203f2bd3ce9,
                    0x1f3698daa8af2ade,
                ])),
                Felt::new(BigInteger256([
                    0xcb0f40dd0ca1dbef,
                    0x6a34c8a81aef90ad,
                    0x6e4fed24452aff3e,
                    0x09c904129b40bd17,
                ])),
                Felt::new(BigInteger256([
                    0x033ce44e4225d357,
                    0x27bbffa25d76d432,
                    0xc72cc12886d53498,
                    0x380b55a4610be0c4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf0dabdaaab4ff56a,
                    0xcc6ab7354d435e48,
                    0x9bfa3469758b8cbd,
                    0x26e462522521dc53,
                ])),
                Felt::new(BigInteger256([
                    0xd4321b0f9f121399,
                    0x5be70868b0a6e39c,
                    0x48e56338b3e91098,
                    0x2814df9473a40b8f,
                ])),
                Felt::new(BigInteger256([
                    0x87a2b2f99f19f946,
                    0x776a115450815c3a,
                    0x526016c4fdefc0e3,
                    0x23fbfc6671c54634,
                ])),
                Felt::new(BigInteger256([
                    0x011c562e7591593f,
                    0xabc48ee6e372d3cb,
                    0xf8f5dee5046aae8f,
                    0x26cc3fddb0a98c0c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x33a2d602bd3ad18a,
                    0x46036cc3dfb7cd67,
                    0xfd82b6c253e66eb8,
                    0x1025d946e242ff26,
                ])),
                Felt::new(BigInteger256([
                    0xf602691d7f2556be,
                    0xd62fda7056958edb,
                    0x07f38b9196377e76,
                    0x05d36eaf008c858e,
                ])),
                Felt::new(BigInteger256([
                    0x1c8946b0f1b83781,
                    0x7cde336769c023a2,
                    0xcb965bc8e8731045,
                    0x0ee55b7b3ad5ccc6,
                ])),
                Felt::new(BigInteger256([
                    0xa2f38a83cb9e4bd7,
                    0x3702d5b0deac7688,
                    0x5d842f706aab049a,
                    0x247cc2c45eb6a74b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x45eaae106dcc48e4,
                    0x54fae907e3c4396d,
                    0x054be5e246e2d669,
                    0x2160e7ac9b2962fb,
                ])),
                Felt::new(BigInteger256([
                    0xc5293b044faf9475,
                    0x288ea64f00331321,
                    0x6d4a0203ab0ff3f4,
                    0x09dcb84a745c66b3,
                ])),
                Felt::new(BigInteger256([
                    0xc6c6fc6bdd141eee,
                    0x1706e6f1d938d9bf,
                    0x34a9cf942269063b,
                    0x20f1e492027e454b,
                ])),
                Felt::new(BigInteger256([
                    0xbe346cc99a8ac66f,
                    0x2a933342f227e99d,
                    0xf4bfaec129d5a184,
                    0x0843c4de0723747b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4a2e551fc959103b,
                    0x83a9dbfd3c0d1593,
                    0x9e15bf1c8b8a6336,
                    0x1d9c618b3b53c605,
                ])),
                Felt::new(BigInteger256([
                    0x680fe37aebb7b787,
                    0xe8add7a41e5fec83,
                    0x19b4c094e0e107e5,
                    0x2ed949154d19792b,
                ])),
                Felt::new(BigInteger256([
                    0x3fec301a2e0f8b68,
                    0xc08ab6314454a14a,
                    0xc9097f4a689fc541,
                    0x0db96dfae62050d1,
                ])),
                Felt::new(BigInteger256([
                    0x641d0475f5eb6efb,
                    0xc29dc93335cb141d,
                    0x460c7a1d29151597,
                    0x1ff486750a52cf6a,
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
                Felt::new(BigInteger256([
                    0x0a7e7c3e99999999,
                    0xb83c0a9bfa6b6a89,
                    0xcccccccccccccccc,
                    0x0ccccccccccccccc,
                ])),
                Felt::zero(),
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
                Felt::new(BigInteger256([
                    0x64b4c3b400000004,
                    0x891a63f02533e46e,
                    0x0000000000000000,
                    0x0000000000000000,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3dbed2378e7866c3,
                    0xa3a216bbb8b9cf8d,
                    0x2aee4c31d40937d3,
                    0x074212f8b9cbe00a,
                ])),
                Felt::new(BigInteger256([
                    0x0599bb3199e63bd1,
                    0x10cf8d45a2fcadb7,
                    0xbd5a95eae9f1ad20,
                    0x05deaa6f24a7dfc3,
                ])),
                Felt::new(BigInteger256([
                    0xb3484385824e9667,
                    0x80c38d519a12e454,
                    0x89754b1e1971de93,
                    0x1bcaf54c4165ba52,
                ])),
                Felt::new(BigInteger256([
                    0x54cd90befe2f6f06,
                    0xe7206901bab65532,
                    0xdd3d868526681dbe,
                    0x13b8ebb23b90e2ee,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc75f6912310fa6cc,
                    0x859fd4d8350a2d92,
                    0x59e68340b984d5bc,
                    0x134406be76078092,
                ])),
                Felt::new(BigInteger256([
                    0xe4faddecbcfcbbcc,
                    0xebee15cb9a369dc8,
                    0x4fba74e5d49a21e1,
                    0x052615f778e9a58e,
                ])),
                Felt::new(BigInteger256([
                    0xe2298e95f307ad56,
                    0x39286c7fe34bd364,
                    0x1e304e6236fb1192,
                    0x13226f4a921db42e,
                ])),
                Felt::new(BigInteger256([
                    0x016ac9c244a58ccd,
                    0xd20817710d444513,
                    0x8798cc49e6ddfdfd,
                    0x126b9cba16aef130,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5ae7b65349774473,
                    0xb98b90a43143f934,
                    0x3d52e92428056488,
                    0x2724704576b0c7f5,
                ])),
                Felt::new(BigInteger256([
                    0x8ae0d456757134fb,
                    0xc934182c7be917a6,
                    0x9942c331d714fc38,
                    0x2ef29e4546ebeadf,
                ])),
                Felt::new(BigInteger256([
                    0xcc9ff08fb2e8c983,
                    0x4cd516123573a62e,
                    0x83c0063036334209,
                    0x293ab98ae35908b3,
                ])),
                Felt::new(BigInteger256([
                    0xc222ad3b8b4e2b4a,
                    0xe1e9dc8bb9f12443,
                    0x93e3f9a8055952e2,
                    0x3d839508cba01f11,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa31e8d4c77293c41,
                    0xaeab5b6f084499ab,
                    0x2aff0f8503eeaea4,
                    0x24a2dbb3164ae7d6,
                ])),
                Felt::new(BigInteger256([
                    0x501a9105ac3eb4ea,
                    0xee9ff65ab48f60ef,
                    0x1b34f05d80a5b814,
                    0x3792697e7cfd7588,
                ])),
                Felt::new(BigInteger256([
                    0x77b01f63fd0269b7,
                    0xa90929d753f99514,
                    0xff79d1437d23c7b6,
                    0x1da29d03b790779e,
                ])),
                Felt::new(BigInteger256([
                    0xcb638fc97b402a69,
                    0xf4b01a5153d7760a,
                    0xa769e3aa0b154def,
                    0x324d2a1eb216fe84,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x17a447d8b16540fc,
                    0xfe83e33a812bd2eb,
                    0x5ec94230b6171051,
                    0x078002cb77fa13e0,
                ])),
                Felt::new(BigInteger256([
                    0x2e62ffde7d2a8521,
                    0x59696261bdc48305,
                    0x7ec9c6d0805b55b9,
                    0x224db24a13f3a6c5,
                ])),
                Felt::new(BigInteger256([
                    0xfee046e8daf7390b,
                    0xbe64519091c01750,
                    0xb37133d358f50bc5,
                    0x31884ad7683e3ebd,
                ])),
                Felt::new(BigInteger256([
                    0x937b6b59e9d6f3ac,
                    0x011f8cf77c156234,
                    0x9eb688d190e0500a,
                    0x1148feace945f827,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb8d0427b61264477,
                    0xc5ab23e056810cc2,
                    0x73cb84e49606761d,
                    0x0586df58ed4256a8,
                ])),
                Felt::new(BigInteger256([
                    0x1d354ba13d7d7f51,
                    0xafdd5f211746ba92,
                    0x7b0458ec51868891,
                    0x16e296a686474a93,
                ])),
                Felt::new(BigInteger256([
                    0xf52aa079ccaa3924,
                    0x3c203cfc56e69edb,
                    0xf8b116154b50e935,
                    0x2a0176b2ac4c1ecd,
                ])),
                Felt::new(BigInteger256([
                    0xb9d10baebd922d09,
                    0x555cbbf2bbbd5da5,
                    0x031114c93d89991f,
                    0x1172a6ed3bbe299d,
                ])),
            ],
        ];

        for i in input.iter_mut() {
            AnemoiPallas_4_3::sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
