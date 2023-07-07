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

/// An Anemoi instantiation over BN_254 basefield with 2 columns and rate 3.
#[derive(Debug, Clone)]
pub struct AnemoiBn254_4_3;

impl<'a> Anemoi<'a, Felt> for AnemoiBn254_4_3 {
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
                    0xddf3700e00578d0a,
                    0xfb78df4f3fcc889f,
                    0x27b578103950b8cc,
                    0x21f157bf072493e3,
                ])),
                Felt::new(BigInteger256([
                    0x61165c7149743bc3,
                    0x0150ef4e0518289f,
                    0xb84333ca59da35cc,
                    0x165ab1d68e2a544c,
                ])),
                Felt::new(BigInteger256([
                    0x2772e438896fb216,
                    0x3632b11cb7384d49,
                    0xb24a366803c89ce5,
                    0x067249d5702c1b61,
                ])),
                Felt::new(BigInteger256([
                    0xcc020921799cca03,
                    0xdfd99cf683433c49,
                    0xb219f1f815dd71c1,
                    0x14b32f407aaba8e7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6783ecf5fc0943e3,
                    0xc4cad5e259e51187,
                    0xeec9f2c7727af272,
                    0x24c998718bf0d00b,
                ])),
                Felt::new(BigInteger256([
                    0xb99c2c764ddf6f5a,
                    0xfa548003f1315357,
                    0x7689a0942e7634c2,
                    0x11aeefdad90cd966,
                ])),
                Felt::new(BigInteger256([
                    0x03174a861a76c455,
                    0xb626fa21ba8bce8c,
                    0x67ffbc20802a6dd8,
                    0x2b86bc43fbc752e7,
                ])),
                Felt::new(BigInteger256([
                    0xfd6f34547b30c0f6,
                    0xc2a387be532687f6,
                    0x948dbd23c99b09ca,
                    0x0dc23f02985d6025,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe7f2d2b21725f3cf,
                    0xe6f34208aed5010f,
                    0x8e599e301347cc80,
                    0x2c72cfd7481ad7ae,
                ])),
                Felt::new(BigInteger256([
                    0x8003859d22641c54,
                    0x8d80411e6d3c7d01,
                    0x075b33da2b41c5cf,
                    0x2168a47f94f217c4,
                ])),
                Felt::new(BigInteger256([
                    0xe7ba6e10ac5a9b82,
                    0xbb734d20e4f42489,
                    0xecfcc5f260590865,
                    0x0b90decf4f8cf89d,
                ])),
                Felt::new(BigInteger256([
                    0x12b5fef117116ae0,
                    0xd5e1740f121425b3,
                    0x8629454ab29652df,
                    0x13a1ea7a99bcc3e5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0b959e70929bb464,
                    0x789619e77029dea6,
                    0x14a887bd952b3f9e,
                    0x0d616f5d2f15e504,
                ])),
                Felt::new(BigInteger256([
                    0xfaaaeea8b20be9e5,
                    0xd6f68d21e82adac8,
                    0x03562d37b4dd7c64,
                    0x0d56f4b627452830,
                ])),
                Felt::new(BigInteger256([
                    0x850e5a31cff2af3b,
                    0x738199f51375af30,
                    0xa6bad9d477002ad1,
                    0x2f888504c9ac6a1b,
                ])),
                Felt::new(BigInteger256([
                    0x2e5af029a63d42dc,
                    0x8e8e605e685d4f13,
                    0xa5e5fe768367c34f,
                    0x087be12a980b728f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x96e0148d6d527388,
                    0xd3011f4846807e03,
                    0x33d9a34d3a02a2c6,
                    0x11e4fa3f552aabd4,
                ])),
                Felt::new(BigInteger256([
                    0x57dc16ac4a0b6f6c,
                    0x325c0e82652b2f70,
                    0x0ae410f2ce57bf76,
                    0x050c9ff42c53fba7,
                ])),
                Felt::new(BigInteger256([
                    0x6d6426dd919f1573,
                    0x0cf6cb660c89076e,
                    0x160dd200adbec090,
                    0x0d9a02751e57df85,
                ])),
                Felt::new(BigInteger256([
                    0x897ab7750da59775,
                    0x26c22b3ce94591f2,
                    0x399e27b04033a108,
                    0x21c3066b56a4f374,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5baf068feaed2750,
                    0x341d1e730593289d,
                    0xa232a1d8b803ed29,
                    0x2d6439fa4317d4ff,
                ])),
                Felt::new(BigInteger256([
                    0x1554b0408b321e96,
                    0x7b7ec979736c4a11,
                    0x9383e33e75bb2e80,
                    0x2ebcea95dc240713,
                ])),
                Felt::new(BigInteger256([
                    0xac5517b865c818c8,
                    0x36e0bac7dfdffad7,
                    0xa08f6768da6b27f4,
                    0x23908195927133ae,
                ])),
                Felt::new(BigInteger256([
                    0x70515be02187fe54,
                    0x19158b8b23744f23,
                    0xd8bc46ba885de88e,
                    0x03157e9926e8baef,
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
                Felt::new(BigInteger256([
                    0xafd49a8c34aeae4c,
                    0xe0a8c73e1f684743,
                    0xb4ea4db753538a2d,
                    0x14cf9766d3bdd51d,
                ])),
                Felt::zero(),
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
                Felt::new(BigInteger256([
                    0x68c3488912edefaa,
                    0x8d087f6872aabf4f,
                    0x51e1a24709081231,
                    0x2259d6b14729c0fa,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfb50158db48bb203,
                    0xf671e7523ffb520d,
                    0x5f6c8a08d958414e,
                    0x1e503a31e268abfc,
                ])),
                Felt::new(BigInteger256([
                    0x71c9086682f8b75b,
                    0x8e9272e573e997c9,
                    0x7e4d2a2e7516ae7e,
                    0x1aeb2a2ee9d97ae5,
                ])),
                Felt::new(BigInteger256([
                    0xbae4b5779dff4bca,
                    0x3d7e065bccf5ea40,
                    0xf67a80830f7824cb,
                    0x19fded6bccf2c4af,
                ])),
                Felt::new(BigInteger256([
                    0xdbcaa8fd90a055df,
                    0x4090f0566eaa4723,
                    0xbf1ef18c7364c9b0,
                    0x23b9fd32744068e6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd0e7d9c5c25eb800,
                    0x1ed103e81f7af479,
                    0xf8626c8791c69996,
                    0x04562eb2c57ac40b,
                ])),
                Felt::new(BigInteger256([
                    0x489b4af413f209c4,
                    0x57a667679dca005e,
                    0x17f2e608131c16e4,
                    0x1db85164c462fdb9,
                ])),
                Felt::new(BigInteger256([
                    0xcd76f31d31dbc69e,
                    0x8336d9a4908eef98,
                    0x7b1f3cbba72f8c03,
                    0x075df1d248462325,
                ])),
                Felt::new(BigInteger256([
                    0xc005de1841957d3d,
                    0x54bf5b7bd0a5d764,
                    0xe8f9717484af7e47,
                    0x15f0fc11b3f680ab,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6b89ef39c856185b,
                    0x40c4d26b416652a8,
                    0xf2019e233b902666,
                    0x0ad1050dac326ecb,
                ])),
                Felt::new(BigInteger256([
                    0x18d4f86f211ced1e,
                    0x1224b3543b866f36,
                    0x840d6ed908fdad3d,
                    0x12d303443f3be609,
                ])),
                Felt::new(BigInteger256([
                    0x9e621723f711f94b,
                    0xaf8225e07a9dc5a5,
                    0x2a41abd4cb39acc7,
                    0x164d69c69f0b2a16,
                ])),
                Felt::new(BigInteger256([
                    0xa6139aac3d40b450,
                    0xd891830477297b40,
                    0xd1062e7f27e60db7,
                    0x04fe96884de4699d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb42d98667ed85023,
                    0x4bf653c0201bd827,
                    0x91d0a00ce6919b24,
                    0x1320b4ccb9a8114a,
                ])),
                Felt::new(BigInteger256([
                    0x85b43a349113d48d,
                    0x751602486a6802d5,
                    0xfa6ed4273991f88c,
                    0x11daab2787a24049,
                ])),
                Felt::new(BigInteger256([
                    0x5a269cf1b99bb678,
                    0x847834e26b7d59cd,
                    0x64e2b26b03e221c8,
                    0x0b050a7c5975cbf0,
                ])),
                Felt::new(BigInteger256([
                    0x29d462adba43bf1b,
                    0xbc0b63aad4ae212e,
                    0x6cb54d3bddec910a,
                    0x1f51d88735141fd8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x49a3dd07ab33003b,
                    0xcb5f0586c4670df5,
                    0x97416ce5fb0837a5,
                    0x202409e895a01c95,
                ])),
                Felt::new(BigInteger256([
                    0xf27223e5fe5dbcaa,
                    0x2aff3ae415f2c128,
                    0x77855fe88ab65c6f,
                    0x0e79349dcd631ec3,
                ])),
                Felt::new(BigInteger256([
                    0xea8d4286ad3bfd37,
                    0x58eee1a410be81b0,
                    0x6e499bcf0ad0920d,
                    0x028b52a5c93ea0e0,
                ])),
                Felt::new(BigInteger256([
                    0x5597dda5076e9120,
                    0x2efe878078be734e,
                    0x53909b7d86e68926,
                    0x141b6d80cd54e75f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xba3e8cfb28ff0af4,
                    0xbffe0e7087fd4063,
                    0xac01cc8f828bccd3,
                    0x25dacb628899a35a,
                ])),
                Felt::new(BigInteger256([
                    0x9b30e902a8922ebf,
                    0x5ff1bedc7761e56e,
                    0x9c75182225195046,
                    0x16435aa28058d558,
                ])),
                Felt::new(BigInteger256([
                    0x58aadb5a9c5f0ef4,
                    0x4cfb06a660a991de,
                    0xef5f761ddaf95f3d,
                    0x00d54ca2a074520f,
                ])),
                Felt::new(BigInteger256([
                    0xa1f6b0a8991f2de5,
                    0x688ce135c9484341,
                    0xc4eab4fd96365bb8,
                    0x1cc8e2b5b19dcd0a,
                ])),
            ],
        ];

        for i in input.iter_mut() {
            AnemoiBn254_4_3::sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
