//! Implementation of the Anemoi permutation

use super::{sbox, BigInteger384, Felt};
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

/// Function state is set to 2 field elements or 96 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 2;
/// 1 element of the state is reserved for rate.
pub const RATE_WIDTH: usize = 1;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 1;

/// One element (48-bytes) is returned as digest.
pub const DIGEST_SIZE: usize = RATE_WIDTH;

/// The number of rounds is set to 21 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 21;

// ANEMOI INSTANTIATION
// ================================================================================================

/// An Anemoi instantiation over BLS12_377 basefield with 1 column and rate 1.
#[derive(Debug, Clone)]
pub struct AnemoiBls12_377_2_1;

impl<'a> Anemoi<'a, Felt> for AnemoiBls12_377_2_1 {
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
                Felt::new(BigInteger384([
                    0x1586e378d7065336,
                    0xe51668bf90dcbd4f,
                    0xa02ddcb1ff06f3ec,
                    0x1d3c2c19d524e0e5,
                    0x27686446d1d62e31,
                    0x00adb6131abd4d08,
                ])),
                Felt::new(BigInteger384([
                    0x0aff9898091ffb98,
                    0xfc3d2d2d7cb7b51b,
                    0x5e33e51f9a396d65,
                    0x6c9650a24f348f61,
                    0x54bdf0fed321cf51,
                    0x000e2c88457a7e21,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xb843088859f4f528,
                    0x067fc1482b55b34b,
                    0x60b3c8fa17b96350,
                    0x28a64f0f9d0b918b,
                    0xc0d69b46f9de6eab,
                    0x00301a2dda213a66,
                ])),
                Felt::new(BigInteger384([
                    0xaaf9ea9673e7312c,
                    0xabbedf8a634f2a49,
                    0xbb2f53353a87d185,
                    0xad2a7fa7ad5a7c2b,
                    0xb2fc67788fb8ffb9,
                    0x014d35b6b40505dd,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x0e4fc8701baab5b0,
                    0x58076fce98bb1dc4,
                    0x3904a3ea9295354f,
                    0xb83a3450aa5f7a44,
                    0xa361dfe34839fcaa,
                    0x0120287ad2fcba13,
                ])),
                Felt::new(BigInteger384([
                    0xd238be36c0278d77,
                    0xfd3ebf209170dfcd,
                    0x539f971d669ead60,
                    0x2a126f427161a070,
                    0xbca8c4ec6523e290,
                    0x002dae4a872b5c06,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x1c01c546a2633bea,
                    0x42ccd67104a4e169,
                    0xa73dc50f638908ec,
                    0x6083a997022c0345,
                    0x076485047d945817,
                    0x004d10729697ca92,
                ])),
                Felt::new(BigInteger384([
                    0x674aa98f943f77c4,
                    0xc0ce63abf275854d,
                    0x27afa36a641e045e,
                    0x8bedbdf5b6b45bcc,
                    0x7d27a2beb54d13ab,
                    0x00e30e5310ac86a7,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x2d8470cb4ee45f4c,
                    0x384522a61c4573a0,
                    0xfb69f3e384e7c1b6,
                    0x3e5bb6c37a524409,
                    0x730cb9c4a4992fda,
                    0x003c20087d555c03,
                ])),
                Felt::new(BigInteger384([
                    0xbb47fc9e2d129ad4,
                    0xdcd3edbce2b63a50,
                    0x214192ad1b78253d,
                    0x2899464c0f5aff60,
                    0x4abffc9403edb774,
                    0x01343420801028ba,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x3477ebc57026e5b5,
                    0xc3eedaa2570dd199,
                    0x29f6a6a3ab302298,
                    0xf023298dfb260449,
                    0x214aaff215ad7c6d,
                    0x00c57098ea9d2136,
                ])),
                Felt::new(BigInteger384([
                    0x84f1f9be9be83817,
                    0xfb33cbe181149adf,
                    0x7f1f26bc64d205f4,
                    0x126191e4f60c135f,
                    0xffb20dfae7ed09b7,
                    0x0167a09be59b728c,
                ])),
            ],
        ];

        let output = [
            [
                Felt::new(BigInteger384([
                    0x56dcddddddddddd4,
                    0x2db2015f37777772,
                    0x8a5a595c4be8b110,
                    0x2041bbb36e056126,
                    0x7e422da67ad9b5fd,
                    0x007c276e8cf025e2,
                ])),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger384([
                    0x59b3fa50b6287770,
                    0x8d8cc2314dc632d6,
                    0x29b1a14daf4e27f8,
                    0xedb8547b775eb818,
                    0xd8e661a68b195234,
                    0x0105b860faad5b24,
                ])),
                Felt::new(BigInteger384([
                    0xa578190a443f6866,
                    0xba8dc4fbbf9b29b6,
                    0x7eb0b8ad3adf4bee,
                    0x878939a768b75022,
                    0x5a0c806ddc201082,
                    0x00b44682633de315,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x3f83be968f6712f4,
                    0x7a14ed2c157a3e9f,
                    0x13379f76f22125ab,
                    0xd2b0fd505a8223b8,
                    0x90580d58fcb34c3f,
                    0x0073a0b34fa4fca9,
                ])),
                Felt::new(BigInteger384([
                    0x6af47e2abade4fc8,
                    0x1a826133f52d710b,
                    0x6c3926ec49ea6ba8,
                    0xbfd567223db99dfb,
                    0xca056d88e9e47a82,
                    0x0036eb7a315a0db0,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xea911dddddddd44f,
                    0xce8327424777727f,
                    0xe774a906518e4834,
                    0x527cf56b51022fb4,
                    0x6e646cdc5f7b965d,
                    0x00eb6a2e45f61af1,
                ])),
                Felt::new(BigInteger384([
                    0x823ac00000000099,
                    0xc5cabdc0b000004f,
                    0x7f75ae862f8c080d,
                    0x9ed4423b9278b089,
                    0x79467000ec64c452,
                    0x0120d3e434c71c50,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x84e9bdfd865e5089,
                    0x0eab1d3400458b74,
                    0xa74a78f9cebc5750,
                    0xa0987466f33c0d1a,
                    0xa1d77beffb04e10c,
                    0x00f619ef9ddfb857,
                ])),
                Felt::new(BigInteger384([
                    0xa52332c502e8f2b2,
                    0xb00092cd1a81d674,
                    0x02bd99a257d157cd,
                    0x4e4e66224eb50d34,
                    0x490a41d0a8f0af0a,
                    0x00c9d0b091fc41e7,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x786ff5f457d0ecd9,
                    0xfae4abe3c01bec25,
                    0xadd9107f0b00bb59,
                    0x68ecd0edbce2e9b4,
                    0xe3e7418bcfe6df03,
                    0x0198bc0843aa45d0,
                ])),
                Felt::new(BigInteger384([
                    0x5a9edfe74f1a3a3e,
                    0x7de5b62af61bc2f8,
                    0xce5dbd562ad9d9af,
                    0xab34a2a4ea35516d,
                    0xb3232c61f92db5af,
                    0x01819a8d709cd99c,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x7ee8d8d054391c63,
                    0xe6fa70920f4a23ae,
                    0xd6f3c78e60216b15,
                    0x46081f64c6e55ccf,
                    0x52d22d34971c69b2,
                    0x0067234b2c5992ed,
                ])),
                Felt::new(BigInteger384([
                    0xc3d853c81dd63cfd,
                    0x0d8c51e01fa15b65,
                    0xb8ac74113761a05c,
                    0x6947fd7c6b38841b,
                    0xafa252ad90fad31c,
                    0x0157b6b8e79d80f0,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x40210e03efbcf044,
                    0xdc4d424546f8105f,
                    0x9c12d2fe2e8df23c,
                    0x7fc44d6b14cad81c,
                    0x79032682da343e51,
                    0x0005aa61f13d1f92,
                ])),
                Felt::new(BigInteger384([
                    0x40f45d409063a560,
                    0x59e94df05e1d0af0,
                    0xef2acd01cb46f018,
                    0xe9d078b7c38e4c21,
                    0x5b2bf7f316fe692d,
                    0x00dcd42aba8cb31f,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xaf134b54892dd66a,
                    0x4d252476e4d3d808,
                    0xe1dcc1b00b2b3c5f,
                    0x85883dec47566551,
                    0xa55551b061956f17,
                    0x003b53b315e5bd4b,
                ])),
                Felt::new(BigInteger384([
                    0xe1de592d0226e734,
                    0xe7a29bf432a04950,
                    0xd297274c627a8516,
                    0x6c2b4308b6c9997a,
                    0xa7e6c2e743ba9f27,
                    0x013b20924a407f22,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x9017a550ad04789c,
                    0xd275e01065eea50b,
                    0x55ca9d4daf222657,
                    0x3e8837e81c8ce31d,
                    0xb07932859e50503e,
                    0x0108d3ad354bde61,
                ])),
                Felt::new(BigInteger384([
                    0x435f90274d805a56,
                    0xeb7ac832e8ba8b8e,
                    0xa808155693c484e8,
                    0x5f46e62878fee442,
                    0xf3b050434ca89302,
                    0x006508ff23925784,
                ])),
            ],
        ];

        for i in input.iter_mut() {
            AnemoiBls12_377_2_1::sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
