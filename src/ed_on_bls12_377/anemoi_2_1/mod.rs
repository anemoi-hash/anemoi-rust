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

/// The number of rounds is set to 19 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 19;

// ANEMOI INSTANTIATION
// ================================================================================================

/// An Anemoi instantiation over BLS_12_377 scalarfield with 1 column and rate 1.
#[derive(Debug, Clone)]
pub struct AnemoiEdOnBls12_377_2_1;

impl<'a> Anemoi<'a, Felt> for AnemoiEdOnBls12_377_2_1 {
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
                    0x07e9a6981bc64437,
                    0xff6ab04ee232a314,
                    0x91f2d1e473c39b02,
                    0x021dfd98a0a69eb2,
                ])),
                Felt::new(BigInteger256([
                    0xa6ab21c48f401e51,
                    0x4b40acdab7589ecd,
                    0x956440dbd76387f0,
                    0x06f370939edd5561,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf02b90f33fe960d0,
                    0x9c97b25e4c7cc2b5,
                    0xca7c097c1c1f7497,
                    0x0b08197fd576ffac,
                ])),
                Felt::new(BigInteger256([
                    0xa06e4bbd846dbae7,
                    0x63fd128907853647,
                    0x7cd5bea332ea8dc2,
                    0x0ca1eee91a060083,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf550c2a2bbe3b67f,
                    0x27ae4eb41ad0207d,
                    0xf06fd31cd32ca1e8,
                    0x07f4cd98926a9e94,
                ])),
                Felt::new(BigInteger256([
                    0x83880fda945552d0,
                    0xe467226eca5a9392,
                    0x3f7d452a76a8dd1c,
                    0x10ec7b35f9d3f2bd,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe241c92d63ba17cf,
                    0xa38f902f21887eeb,
                    0x32e2d05a6e53a407,
                    0x0e494be3d6aea2be,
                ])),
                Felt::new(BigInteger256([
                    0xb3338ae1cc24e0a3,
                    0xd4fa55aa5aee890d,
                    0x987d56b34dc2a06f,
                    0x0eaa6878d75749c3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0bda9740913c22f7,
                    0x105ce31fca8248bc,
                    0x9d32da44eb944201,
                    0x06647632051af9a3,
                ])),
                Felt::new(BigInteger256([
                    0x5ac7ceba7a0c8d91,
                    0xbf5b01b784b13f3f,
                    0xbae4f764ec12771e,
                    0x09dc9e08ec4d6ea9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7954da45193d45d3,
                    0xea3ed9505750dcef,
                    0x35acf18b0eb4bd79,
                    0x0caf7de01cc5cc64,
                ])),
                Felt::new(BigInteger256([
                    0x365cbdb6ccdddd4f,
                    0x9755977fd462bd9a,
                    0x7023b3a44fa51013,
                    0x08737a611d65e1fc,
                ])),
            ],
        ];

        let output = [
            [
                Felt::new(BigInteger256([
                    0xb76f9745d1745d17,
                    0xfed18274afffffff,
                    0xfce619835b36a173,
                    0x068b6ffd78dc8d16,
                ])),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0x547eaa4fe92c17b5,
                    0xdbdc56d3eeee3424,
                    0xd4e029a17d9b9417,
                    0x0e71b0e9185f0412,
                ])),
                Felt::new(BigInteger256([
                    0xd66a1b1b52ed0c94,
                    0x7350e41fff172b46,
                    0xdee8ec7bf3d076d4,
                    0x08092cd96d87788d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbc9fbddce41f262d,
                    0xdd9574526d0633f2,
                    0x74ab4b549cae7cc8,
                    0x00f4b6647a675f0c,
                ])),
                Felt::new(BigInteger256([
                    0x86ef9156cec705ed,
                    0x31911e672304df9a,
                    0xdb1cd013ebe897a2,
                    0x110866f6f86e425d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x53e71745d1745bdc,
                    0xaa1116eabffffeb8,
                    0xff0b3527e2b10fca,
                    0x0da5b495c3ed1bcd,
                ])),
                Felt::new(BigInteger256([
                    0x8cf500000000000e,
                    0xe75281ef6000000e,
                    0x49dc37a90b0ba012,
                    0x055f8b2c6e710ab9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe9bd80afd48334f6,
                    0xb08ba3f89193de0e,
                    0x60fe916bb1314021,
                    0x0fe6ede65d5573a9,
                ])),
                Felt::new(BigInteger256([
                    0xbba99b953a5166c8,
                    0xe0c5bb74fb426643,
                    0x11c8e3721280441c,
                    0x0269bce008e98fdd,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3ca8ffaf2fd56321,
                    0xa7008902000f08a4,
                    0xba20d120328112ff,
                    0x0bbedf4d8ed9cdf8,
                ])),
                Felt::new(BigInteger256([
                    0xf8c4355236fe86a8,
                    0x61c522c174ba35b8,
                    0xe4077f36d3d56a6c,
                    0x0babda3faa160297,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x872c9e78bff7b407,
                    0x26bb3f9d26e33d63,
                    0xb5b8ddb6c68ece27,
                    0x0d80fe3865b31dcb,
                ])),
                Felt::new(BigInteger256([
                    0x38d6d325d31ed930,
                    0xabd22218b74067a7,
                    0x03a23ed3137b3e99,
                    0x056998182c3a8de6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8e5ea733820f52cd,
                    0x180a4144235f023e,
                    0xa686d3184f7c6374,
                    0x10c2779ca91a6591,
                ])),
                Felt::new(BigInteger256([
                    0xf83fed1a114fd80f,
                    0x8964eba4c978b00b,
                    0xe931662da7827e95,
                    0x0fc72cf4c2c5fcfa,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x664e9d2678aa9b16,
                    0x0243b4525b2bfd9f,
                    0xd05f7b45df65708c,
                    0x02a79b6f39b223f6,
                ])),
                Felt::new(BigInteger256([
                    0x5283bedc9df8caa5,
                    0x5e93e5c6fcdcb3c7,
                    0x50d46dc67507ec46,
                    0x04e263cbcbacb04b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd3035e1ceff53cbd,
                    0x157f7343b128ae12,
                    0xf53089f6ce8a2f66,
                    0x0c33f64ddb82c89c,
                ])),
                Felt::new(BigInteger256([
                    0xeecf610e06d78e50,
                    0x72b392a91ad47573,
                    0xd4a35cd8c0baa80a,
                    0x01e17f923a27ee34,
                ])),
            ],
        ];

        for i in input.iter_mut() {
            AnemoiEdOnBls12_377_2_1::sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
