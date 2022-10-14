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

#[inline(always)]
/// Applies exponentiation of the current hash
/// state elements with the Anemoi S-Box.
pub(crate) fn apply_sbox(state: &mut [Felt; STATE_WIDTH]) {
    let mut x: [Felt; NUM_COLUMNS] = state[..NUM_COLUMNS].try_into().unwrap();
    let mut y: [Felt; NUM_COLUMNS] = state[NUM_COLUMNS..].try_into().unwrap();

    x.iter_mut().enumerate().for_each(|(i, t)| {
        let y2 = y[i].square();
        let beta_y2 = mul_by_generator(&y2);
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
        let beta_y2 = mul_by_generator(&y2);
        *t += beta_y2 + sbox::DELTA;
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
        // Generated from https://github.com/vesselinux/anemoi-hash/
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
            apply_sbox(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }

    #[test]
    fn test_mds() {
        // Generated from https://github.com/vesselinux/anemoi-hash/
        let mut input = [
            [Felt::zero(); 4],
            [Felt::one(); 4],
            [Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            [Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0xe3a67a3ffe11209b,
                    0x10ede71af1ba8a5f,
                    0x9ed2ff2f68ed12ab,
                    0x2584a1ba8305f0dc,
                ])),
                Felt::new(BigInteger256([
                    0x9cf807966a18303a,
                    0xc2f06e628d691dd8,
                    0x352cde754ea4e90c,
                    0x2b6abf15ef4072af,
                ])),
                Felt::new(BigInteger256([
                    0xfe876e4e0ecb5fb5,
                    0xc657c7d246db82fc,
                    0x9406075eb2150382,
                    0x247127688729aecf,
                ])),
                Felt::new(BigInteger256([
                    0xfb8aa15e79bfe938,
                    0xf7855d089b7dcaf4,
                    0x94758324ca397c79,
                    0x22857fb7b1d1a862,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x651fe7bbf2e21e86,
                    0x6558934b3e1a13a2,
                    0x1a4e4af909f4c0b1,
                    0x023a75e5b821ba50,
                ])),
                Felt::new(BigInteger256([
                    0xdf761481bbb032c7,
                    0xd6ad63a27c302360,
                    0xcc840538ab9ea66c,
                    0x2246fa45e4cbd1b5,
                ])),
                Felt::new(BigInteger256([
                    0xffa5b0f50d95e357,
                    0xff7d80e92283feab,
                    0x6a76315ecaa02dd4,
                    0x2699e21a27c1c792,
                ])),
                Felt::new(BigInteger256([
                    0x25b179123755b816,
                    0xb37d9fa90a55f711,
                    0x6972d1ef4456784d,
                    0x0dc1fca2f55ebf54,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9bc6e1fa0103edef,
                    0x46821024a443ba90,
                    0xd933d21eda5c64a4,
                    0x1228900a1e0ff53b,
                ])),
                Felt::new(BigInteger256([
                    0xe69cba30a709e868,
                    0x13d8914737f37513,
                    0xfeeb241f61220b22,
                    0x1d99db1feb0166ae,
                ])),
                Felt::new(BigInteger256([
                    0xa295a5c9ad15f23a,
                    0x2a0a9351b6aa8433,
                    0xe929db5620da715e,
                    0x157fae24cc6f97d8,
                ])),
                Felt::new(BigInteger256([
                    0x2b23eccdfeb03092,
                    0xfa10b8c8f3b5a3ea,
                    0xd6222f73867d9cbb,
                    0x0dfa532f87a7c2a2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9b989db51acec3ba,
                    0xaf517ca61fa9e6b2,
                    0x0d16aa84ac95dd4f,
                    0x14da6e83b4328542,
                ])),
                Felt::new(BigInteger256([
                    0x08d9a5abbf02ca18,
                    0xd8a6d4a29a33653d,
                    0x3301658c30b6f168,
                    0x28020cb710f6139e,
                ])),
                Felt::new(BigInteger256([
                    0x6316357fddee5148,
                    0xec75d6d5cf278eb1,
                    0xd8b321f315aff490,
                    0x1a7d09646c0c6eef,
                ])),
                Felt::new(BigInteger256([
                    0xb9b9e5f0ef3c05f0,
                    0x053bf9f4bb9de260,
                    0x790bb39ac8cdfe15,
                    0x080c50a32366538a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xcbd6e513828a991f,
                    0x7e273b1efccb4fdc,
                    0x0ad151faf56f08c5,
                    0x2eb5224b93e9c63a,
                ])),
                Felt::new(BigInteger256([
                    0xdb647871e5cf59c7,
                    0x312d272549936af3,
                    0xe64b4b2a726f3eb4,
                    0x00c314158019edfc,
                ])),
                Felt::new(BigInteger256([
                    0x1dc7f47cdbef4294,
                    0xebc90f01fd25e291,
                    0x33c82abf59ceb5af,
                    0x1ae60b5d2aa712bd,
                ])),
                Felt::new(BigInteger256([
                    0x029c62ced5dfa996,
                    0x4beef57ebd30e43f,
                    0x659b26466239a483,
                    0x2524d7527f929305,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x276c6b5a72b683ac,
                    0xca3177e68d538cf1,
                    0x2dd841aabcb449cc,
                    0x259d81ac324e1708,
                ])),
                Felt::new(BigInteger256([
                    0x4f15326baba2dc41,
                    0xeab158ddde41ddf3,
                    0x7606e6dd2b3b1b2b,
                    0x0d721de513d497e9,
                ])),
                Felt::new(BigInteger256([
                    0xfd07cb7d239ca596,
                    0xbdcfcc0c68cd3275,
                    0xe62083e96b5a64f2,
                    0x161381d27bec1aaa,
                ])),
                Felt::new(BigInteger256([
                    0x5e22c54fe07b0208,
                    0x4f31efacc3ff7f20,
                    0x441757af9b35930c,
                    0x275ab8ec59c82bb0,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(); 4],
            [
                Felt::new(BigInteger256([
                    0x115482203dbf392d,
                    0x926242126eaa626a,
                    0xe16a48076063c052,
                    0x07c5909386eddc93,
                ])),
                Felt::new(BigInteger256([
                    0x075ac9ee7eccb924,
                    0xc19fb16041c6327c,
                    0x0aad7b8599a48723,
                    0x255b297c2ed174eb,
                ])),
                Felt::new(BigInteger256([
                    0x115482203dbf392d,
                    0x926242126eaa626a,
                    0xe16a48076063c052,
                    0x07c5909386eddc93,
                ])),
                Felt::new(BigInteger256([
                    0x075ac9ee7eccb924,
                    0xc19fb16041c6327c,
                    0x0aad7b8599a48723,
                    0x255b297c2ed174eb,
                ])),
            ],
            [
                Felt::zero(),
                Felt::zero(),
                Felt::new(BigInteger256([
                    0x115482203dbf392d,
                    0x926242126eaa626a,
                    0xe16a48076063c052,
                    0x07c5909386eddc93,
                ])),
                Felt::new(BigInteger256([
                    0x075ac9ee7eccb924,
                    0xc19fb16041c6327c,
                    0x0aad7b8599a48723,
                    0x255b297c2ed174eb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x115482203dbf392d,
                    0x926242126eaa626a,
                    0xe16a48076063c052,
                    0x07c5909386eddc93,
                ])),
                Felt::new(BigInteger256([
                    0x075ac9ee7eccb924,
                    0xc19fb16041c6327c,
                    0x0aad7b8599a48723,
                    0x255b297c2ed174eb,
                ])),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0x062cecbeb2e2b974,
                    0x933af28e60a08442,
                    0x1568c96bd057c4b8,
                    0x1697f3a3ad32686d,
                ])),
                Felt::new(BigInteger256([
                    0x373db5a4d1c66208,
                    0x4d9e70eade671584,
                    0x04c6af4bbca9867b,
                    0x0e69fd1b34746ba3,
                ])),
                Felt::new(BigInteger256([
                    0x7edfd41af5280dc9,
                    0x1b89df5c9f2cbed1,
                    0xdfe70dd3dd75d647,
                    0x2f10590b84eb747d,
                ])),
                Felt::new(BigInteger256([
                    0xc6c5465a64cc913b,
                    0x52712633eb0c5fc9,
                    0x0aca5fb6c5f27d3f,
                    0x2075473272572bcc,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8b410d1374f8bc4d,
                    0xba5de90fe1c6e8aa,
                    0x0f39cf3609ce033c,
                    0x0846c7d1a421ef1e,
                ])),
                Felt::new(BigInteger256([
                    0x4518afa5421d6a67,
                    0x6e45b440b91312d3,
                    0x41e12d24478757c5,
                    0x0ab70347effffee6,
                ])),
                Felt::new(BigInteger256([
                    0xac6173c3af1d678d,
                    0x82f34d41a0fe5dfa,
                    0x3834da9ea1345111,
                    0x20c7060baa40d5b8,
                ])),
                Felt::new(BigInteger256([
                    0x8c88f41269f41f70,
                    0x5954938b349b8381,
                    0xa27435cdab3a704e,
                    0x2826575764210867,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd75bf85e4527ac99,
                    0x5308eed77b3a84b1,
                    0x6554b30ffabfd54f,
                    0x0a2d84841cb0e8f5,
                ])),
                Felt::new(BigInteger256([
                    0x309017349e03f0ec,
                    0x7571f33c4131389c,
                    0x7698f798cfe032b2,
                    0x0bbe1a395fe28165,
                ])),
                Felt::new(BigInteger256([
                    0xd6c452142d7509f9,
                    0xe0af082caf4365f7,
                    0xd94f7bbf678b9878,
                    0x1e150f2b0bc4ea03,
                ])),
                Felt::new(BigInteger256([
                    0xaea183d8847b1597,
                    0x9d14d6b4f3912100,
                    0x0477c327547a8a0d,
                    0x0ef63ec02d5b1591,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3de4768aa6dd2774,
                    0x0a43256b1d60814f,
                    0x357a4fbc3bb800cf,
                    0x2c17f7c324b17fc9,
                ])),
                Felt::new(BigInteger256([
                    0x0e2565072a23489f,
                    0x30ec052fb8ff8983,
                    0xaa7f839d5f5aeabd,
                    0x1b1d08a7db75b27c,
                ])),
                Felt::new(BigInteger256([
                    0xa6dbfa59b089fc81,
                    0x331c13e4c0a2c3e7,
                    0x4ad4d3bd885c836a,
                    0x271f1e5d865a0030,
                ])),
                Felt::new(BigInteger256([
                    0xdf690c5f3e924c3d,
                    0x56c73d61402c454d,
                    0x489111beabc2ce14,
                    0x2f11c7973cb72f2d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x21e3c2525b7ba92d,
                    0x7a2d45fd7113c62b,
                    0x0562edc3cb3b6c84,
                    0x009a10193305f007,
                ])),
                Felt::new(BigInteger256([
                    0x410fbf68f842554e,
                    0x9fb4f91d9ccebd75,
                    0xf6741475d4218441,
                    0x02914461192bbe11,
                ])),
                Felt::new(BigInteger256([
                    0xe3b32817b8b376c4,
                    0xe0474d61e3bef6d7,
                    0x90531b176ca314d7,
                    0x150e5c843d248ae9,
                ])),
                Felt::new(BigInteger256([
                    0x8cc0e0ad2d8ca999,
                    0xf51d8c963ff0fc8b,
                    0x2c71364f1e369bd9,
                    0x29acd27700e31350,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd88b76869d221b28,
                    0xf2c417eebfa75c3d,
                    0xd79cb08bbce442f2,
                    0x1d8f8ce88c9a3e9a,
                ])),
                Felt::new(BigInteger256([
                    0x60767dd1d20f332b,
                    0x93facb874c545d92,
                    0x8c3c6d135ee53349,
                    0x055827b8f7401366,
                ])),
                Felt::new(BigInteger256([
                    0xdcf90f999a56f83c,
                    0x599e7eaf2d838167,
                    0x85d857feda421129,
                    0x08cca17e0b293b5d,
                ])),
                Felt::new(BigInteger256([
                    0x57d26e331a249103,
                    0x3329dd8888e5ec20,
                    0xbf59462f789f4011,
                    0x001517d9bc362c99,
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
