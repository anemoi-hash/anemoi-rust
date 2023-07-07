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

/// An Anemoi instantiation over BN_254 basefield with 1 column and rate 1.
#[derive(Debug, Clone)]
pub struct AnemoiBn254_2_1;

impl<'a> Anemoi<'a, Felt> for AnemoiBn254_2_1 {
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
                    0xc5c837622b46627a,
                    0xe6e7b0c7015be949,
                    0x3d38674b6930206e,
                    0x08793a9fbc1d1a51,
                ])),
                Felt::new(BigInteger256([
                    0x7ce30b89b6d38200,
                    0x93e5684f35b5a027,
                    0x43056a6398898666,
                    0x06af236c2905216c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xaee6936bc836e520,
                    0x131bb957415f9e4e,
                    0x748e65806b922907,
                    0x25483032f5433055,
                ])),
                Felt::new(BigInteger256([
                    0x7c195ed10817019a,
                    0xc4fd278aee4e0176,
                    0x91abb9fa59dbb472,
                    0x024c411a3409a344,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x602021b664299599,
                    0x331d392f5f82ef0b,
                    0x46563e33043a71b1,
                    0x2e07fec7e2114a05,
                ])),
                Felt::new(BigInteger256([
                    0xfead181abbfb3ffa,
                    0x7453f9c1ff01e7fb,
                    0xbafb4eedba0eaa9e,
                    0x2c92c4b1a5d04a22,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6162ec3c8683b781,
                    0xc6badca61f360847,
                    0x92225ead597b9cac,
                    0x2989cb3c2d7a43db,
                ])),
                Felt::new(BigInteger256([
                    0x9e8e53ef94da86e4,
                    0x5bc357ac2907e787,
                    0x059662df56742ed1,
                    0x164f41c88da7f857,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x06724e02bea42e74,
                    0x68a2e8108b7f1d3a,
                    0xb7016099a315d396,
                    0x2a3ee79cb68211b0,
                ])),
                Felt::new(BigInteger256([
                    0xfc5cda1cb057df0e,
                    0x635cb756672bdcb8,
                    0x64fd93292610acbf,
                    0x0ff932c159b5a445,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe4c310d9c3cad478,
                    0x4888822d9209d66e,
                    0x3bc86bb6b62428ab,
                    0x04626814f5396f72,
                ])),
                Felt::new(BigInteger256([
                    0x19b45b4c93d15f57,
                    0x4892ee9e7886f317,
                    0x59ed1a786c342cf1,
                    0x24ab86fdeba9e792,
                ])),
            ],
        ];

        let output = [
            [
                Felt::new(BigInteger256([
                    0xafd49a8c34aeae4c,
                    0xe0a8c73e1f684743,
                    0xb4ea4db753538a2d,
                    0x14cf9766d3bdd51d,
                ])),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0xbebb490c2e6f962f,
                    0x465ac2ad04e364b4,
                    0xff4b8beb7ecbd2f7,
                    0x1daa245477c958e5,
                ])),
                Felt::new(BigInteger256([
                    0xe07a0c33042f6fc0,
                    0xfbc9e1489023576a,
                    0xd40daa7093c5a810,
                    0x01eb281368a433ca,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9221c13bf351088f,
                    0x7ba14974192568d6,
                    0x764d83e12590a30c,
                    0x1efdfa43167d3d99,
                ])),
                Felt::new(BigInteger256([
                    0xa8fad3571dcabf05,
                    0x20a54df8cdb610bd,
                    0xb562a9b2221bf6a7,
                    0x113c1d95ad2f3ee3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc1291cac726de779,
                    0x730b09508e12a9ad,
                    0x965495beb3b74a80,
                    0x1c9527fa5aabb1b1,
                ])),
                Felt::new(BigInteger256([
                    0x68c3488912edefaa,
                    0x8d087f6872aabf4f,
                    0x51e1a24709081231,
                    0x2259d6b14729c0fa,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x108354a196d6f5df,
                    0xdbac302dd391fac8,
                    0x06ba3eb06f940e40,
                    0x2234e012022ced7c,
                ])),
                Felt::new(BigInteger256([
                    0x16964980bebbc3e0,
                    0x043dae72d7622803,
                    0x9847ba096bf64c14,
                    0x2004cf4bc9a14f45,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb0801bfb7c8c96fd,
                    0x7ab73e745b84dfe8,
                    0x5d83233fe3e7325d,
                    0x2c17f8135cabb694,
                ])),
                Felt::new(BigInteger256([
                    0xf9562832a846ad62,
                    0x50e1d09d9dadc36e,
                    0x86eb9428f6538c4e,
                    0x00fab2ca058bac6d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x901064b6a3737e9e,
                    0x5799e7aa5f5ba289,
                    0x553c76a0de03fb20,
                    0x0fe8c031717478bd,
                ])),
                Felt::new(BigInteger256([
                    0x541654f90ec2a42d,
                    0x5fe9298c01b7a0a6,
                    0xfb1a11b04ebcf56c,
                    0x2fa25252c7d37cc6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa5faffa4fb80b7d5,
                    0xf96402db92c61e8a,
                    0x3c5cdee50778e58a,
                    0x1a03545b20de7040,
                ])),
                Felt::new(BigInteger256([
                    0x54bb9a99206b2f7a,
                    0xe34e50eb0eb4bd49,
                    0xb594766e2689bae5,
                    0x18f22655ad241b4c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xee15a201d38cd155,
                    0x830071c2b687ae1b,
                    0x1364dceeecb08db2,
                    0x19102e1b8a7c201e,
                ])),
                Felt::new(BigInteger256([
                    0x61dc6943286be407,
                    0x420ae10efb603adf,
                    0x6791ac11acac6a57,
                    0x04a12cbb410b54d4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7a2250c6d796bbdc,
                    0x90375ad998f6c21e,
                    0xfdfa3e39a67a0b6b,
                    0x11dd033c9c1e52af,
                ])),
                Felt::new(BigInteger256([
                    0xcf1fbdf282de1984,
                    0xf52da3aca8e2a241,
                    0x4294a8c6e67aace1,
                    0x2ea4466d05ac8b7c,
                ])),
            ],
        ];

        for i in input.iter_mut() {
            AnemoiBn254_2_1::sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
