use super::{sbox, BigInteger384, Felt};
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

/// Function state is set to 2 field elements or 96 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 2;
/// 1 element of the state is reserved for rate.
pub const RATE_WIDTH: usize = 1;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 1;

/// Two elements (96-bytes) is returned as digest.
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
        let beta_y2 = y2.double().double().double().double() - y2;
        *t -= beta_y2
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
        let beta_y2 = y2.double().double().double().double() - y2;
        *t += beta_y2 + sbox::DELTA
    });

    state[..NUM_COLUMNS].copy_from_slice(&x);
    state[NUM_COLUMNS..].copy_from_slice(&y);
}

#[inline(always)]
/// Applies matrix-vector multiplication of the current
/// hash state with the Anemoi MDS matrix.
pub(crate) fn apply_mds(state: &mut [Felt; STATE_WIDTH]) {
    let xy: [Felt; NUM_COLUMNS + 1] = [state[0], state[1]];

    state[0] = xy[0] + xy[1].double();
    state[1] = xy[1] + xy[0].double();
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
                *r += *s * mds::MDS[i * STATE_WIDTH + j]
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
                Felt::new(BigInteger384([
                    0x409d14f77f773dcb,
                    0xa62d21f921dc98b3,
                    0x737bc8034004baf3,
                    0xb8e35082ccb50e49,
                    0x6d1c3b6f57a1d1c8,
                    0x0172cb855435c4f4,
                ])),
                Felt::new(BigInteger384([
                    0x78377ba8a65a3d38,
                    0xca98a0756754b371,
                    0x8676f0bd65323c02,
                    0x26f614f8e8b671d8,
                    0x84150b8c256a7306,
                    0x00720f44ff79ecc7,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xc60cd47672f3c930,
                    0xc5f8460137985bae,
                    0xe78af7e7da0fa704,
                    0x3f9f10dd432d1e77,
                    0x9051f55e9223f279,
                    0x009a8088396d6e82,
                ])),
                Felt::new(BigInteger384([
                    0xf596857f1440d16c,
                    0x4eaf8e319ad48646,
                    0x4e062c4b7a8b883e,
                    0x12a56b2976bb68df,
                    0x1a2c67666cdc4de4,
                    0x0086b4204a6b9833,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xad80a29695921285,
                    0x782b0499df818c3a,
                    0x359c824fb1c2712e,
                    0x164016e7e1bee789,
                    0x5f50ce67108e803c,
                    0x00f06917a3e4364b,
                ])),
                Felt::new(BigInteger384([
                    0x79c436b66b482e2c,
                    0xe526ce1b7e66619f,
                    0xa0c749939da2e3c7,
                    0xce9a55a0436ada28,
                    0x123e29e2180c9fda,
                    0x015e0682d1e43417,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x0d35ff7af3dd148b,
                    0x87d750e204a07d5b,
                    0x397bebf651846587,
                    0xb36a97eae5ad9e83,
                    0xa27b901a29392887,
                    0x0001fa6df4af5b1c,
                ])),
                Felt::new(BigInteger384([
                    0xcdd2f9e63fbcf324,
                    0x184249a2bd0fd3c6,
                    0xdf8ccf07dc603f18,
                    0xdaa21ac5b99e559c,
                    0xbbe7893e296f0061,
                    0x002d1c34349ad7b6,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x7a5be15e6c29fefc,
                    0x2ed4a10c5b54a269,
                    0x8305153a3fba567a,
                    0x2a6b4ea643a38503,
                    0x128319f226174c76,
                    0x00f3be4239c1d1ed,
                ])),
                Felt::new(BigInteger384([
                    0x03372eb369545d7b,
                    0x1770912978099190,
                    0xcbd35fd81057f999,
                    0x4bc44b11230fd820,
                    0xd5986883aa591888,
                    0x00ff97763ef74bb4,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x3f214de5e1749d9a,
                    0x8a16fb86d7bfb3ef,
                    0xbb105d7dcb47e1ab,
                    0x989b74b88cd717e2,
                    0x71eb825df5e22d59,
                    0x01423afa47cfccb9,
                ])),
                Felt::new(BigInteger384([
                    0x66d26fa9f78b73a2,
                    0x3850c8a3fbd611cb,
                    0x031cda55b5b79c81,
                    0x0734d5db8a5d5e1b,
                    0xad2a379dd5489d70,
                    0x01549edb6bfbc44a,
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
                    0x13fb4947c9dd194a,
                    0x5b95d047a400a365,
                    0xcfe6c034c95effe4,
                    0xa16dac2b28538927,
                    0x341bd6d9a7b370ad,
                    0x000a72f667abf4dc,
                ])),
                Felt::new(BigInteger384([
                    0xb8c04bc77aaf09da,
                    0x922856595695e099,
                    0x2f559b62cac41321,
                    0x6f44ba4814b0d7d9,
                    0xc528967f2e51cd63,
                    0x01341b76eacc48fe,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x60ad5d1504d3d8bc,
                    0x6488aad41a8420e1,
                    0xba0344919abf8d9e,
                    0xd4469f652a19cd1f,
                    0x2bd9e07aa7f42399,
                    0x01678d888cd69bda,
                ])),
                Felt::new(BigInteger384([
                    0x80156064a651b825,
                    0x0ae583f8b71c2f19,
                    0xafd2599ab6d822d8,
                    0x36c046170b0409eb,
                    0x64e3ccb2bb83e1f5,
                    0x0098d85292c7a554,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x732db143cc22b7d5,
                    0xd7e5cb149495cdd8,
                    0x1159c732e085370d,
                    0x67cdd163051dde5e,
                    0x26a93bbf2d0ea1eb,
                    0x01571694c11c6360,
                ])),
                Felt::new(BigInteger384([
                    0x2c9727c2aa0c6770,
                    0xee9e289ad84140fa,
                    0x3b209fa58b0c4596,
                    0x95c7d944ef09ca8c,
                    0x50357841ae826e5d,
                    0x0111745e90d1a857,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x4dab80ca4d29b2c3,
                    0x23d5e2caafaebdf1,
                    0x6ad72e48b4e5dfb9,
                    0x4a2e357f80564573,
                    0x095b8ce0df63757c,
                    0x0079c23259626c8e,
                ])),
                Felt::new(BigInteger384([
                    0x1b95909b94626b38,
                    0xdba5ffbd66f4c805,
                    0xa708b92a92ed2e6a,
                    0x7cc881814c43aa54,
                    0xf0944bdf3bf4d6fe,
                    0x013cc2f1ae36894b,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xec94e749bbe35e92,
                    0x169acf71e8c26deb,
                    0x379d0f5f0517f56c,
                    0x558ed74804762c07,
                    0x993c6e9789c6518b,
                    0x000761a9d8094184,
                ])),
                Felt::new(BigInteger384([
                    0x832edb9df4317b67,
                    0x46c82f9aa1e93781,
                    0x55cd7257eaf3a3eb,
                    0x88bf4e576a333e71,
                    0x313ac5c7161abe81,
                    0x002f81fb3245cf6c,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x52a228b2be44b04e,
                    0x52c397e2ffad07d8,
                    0x538b816f3a7c69c4,
                    0x47be856a1fea2d38,
                    0x372bcbd9a6abf378,
                    0x0046e6728a8058e7,
                ])),
                Felt::new(BigInteger384([
                    0x8c24b10de0442c30,
                    0x641e3cb3d577e646,
                    0xd34849db31810bde,
                    0x27c2c06f8d050c08,
                    0x3ee07fa9b7c3a4e1,
                    0x01423b9dcebcb7f2,
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
                Felt::new(BigInteger384([
                    0xa2381ad0fa85da4d,
                    0xb37ae6c82df1a225,
                    0xa6138295df3e537b,
                    0xbe3f7b811fed363b,
                    0xfd62ad33d22c8628,
                    0x012dc55b79d7a7d8,
                ])),
                Felt::new(BigInteger384([
                    0x39ead92e0b72eaae,
                    0x22f77165198c459d,
                    0xf4d909c1b07f4c05,
                    0x17585472a42ea67c,
                    0xc830227bac7da198,
                    0x0136f2d96f1590a0,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x10b748b317609117,
                    0x9ff974703c01848d,
                    0xd9528382a38b4c9d,
                    0x062a0a279590e432,
                    0xd6c8f3bb3c198ef2,
                    0x01532c828022853e,
                ])),
                Felt::new(BigInteger384([
                    0x61b05ac58a7ca2df,
                    0xf02d22437241e755,
                    0xdaae8cb35b85fa33,
                    0x13cf55cd6e2e3a53,
                    0x741d84a15d401a05,
                    0x0022ad156b996d15,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x9d10de025646d885,
                    0x26642dfaaeb2b3d5,
                    0x045ec79a78414a3c,
                    0x6df033732637e6db,
                    0xa14a7d6d02e90668,
                    0x01aa7b2bb73cf4be,
                ])),
                Felt::new(BigInteger384([
                    0x534126209c2cd7ce,
                    0x5948acac781c7f0e,
                    0x94dbd400ef25cd74,
                    0x42dee69b5a12bba4,
                    0xf7d416d49845d7f6,
                    0x005deb1bf4b1f365,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xbb099774235ee2a7,
                    0x984ec7a4119824a4,
                    0xc054618bef68a913,
                    0x1b64cdfc9b4aae4f,
                    0x53355da7cbff18a6,
                    0x00d5535e05438472,
                ])),
                Felt::new(BigInteger384([
                    0xbfb20df878f96634,
                    0xb9dae5342a281150,
                    0xc604bce27b8b14dc,
                    0x1cc31dd27f454cdc,
                    0xf6b364e3124ad12b,
                    0x000d992e4437cb5f,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xa3244a1f51323cda,
                    0x7668b8fca9055e26,
                    0xead158b1f98ef3e6,
                    0x32be60f23b93f9c8,
                    0xcd8e9cfd948b001e,
                    0x014dbc203fcf9fb5,
                ])),
                Felt::new(BigInteger384([
                    0x7d94f9a75553b28b,
                    0x773921e5f3a59fdc,
                    0x2c243f1e76104bfe,
                    0x566dbd2db0601714,
                    0xbb10f80ccc1a35eb,
                    0x00e000801e2cd172,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x1c271128fa045937,
                    0xfc8b6dd91fbe1c50,
                    0xe9b45b16ab6281e7,
                    0xe9c0e194a92f9f11,
                    0x413de592dd6ae916,
                    0x011ed0c2915dbc53,
                ])),
                Felt::new(BigInteger384([
                    0xdefbc98dae06a04b,
                    0xa223ad7b0f188511,
                    0x29e5be215ba719c7,
                    0x126a2735c229e455,
                    0x7b0d1d07787a6338,
                    0x01ab2103f490d20b,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger384([
                    0x0869fffffffffe38,
                    0xf3c1de8a7fffff13,
                    0xde791afc9f77bfd6,
                    0x71ebc7264b752910,
                    0xe6ddc13e80b58eb9,
                    0x01a83325a8f9ddce,
                ])),
                Felt::new(BigInteger384([
                    0x0869fffffffffe38,
                    0xf3c1de8a7fffff13,
                    0xde791afc9f77bfd6,
                    0x71ebc7264b752910,
                    0xe6ddc13e80b58eb9,
                    0x01a83325a8f9ddce,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x059bfffffffffed0,
                    0xa2813f06ffffff62,
                    0x3efb675314fa7fe4,
                    0xf69d2f6edcf8c60b,
                    0x99e92b7f007909d0,
                    0x011accc3c5fbe934,
                ])),
                Felt::one(),
            ],
            [
                Felt::one(),
                Felt::new(BigInteger384([
                    0x059bfffffffffed0,
                    0xa2813f06ffffff62,
                    0x3efb675314fa7fe4,
                    0xf69d2f6edcf8c60b,
                    0x99e92b7f007909d0,
                    0x011accc3c5fbe934,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x0bfc4d2d116bafa7,
                    0xcb530f0a010a2d5f,
                    0x51ded1b9cc2a5b85,
                    0xb8aa708066605c17,
                    0x014ce6aa51e536e2,
                    0x003f36822878a745,
                ])),
                Felt::new(BigInteger384([
                    0x74498ed0007e9f46,
                    0x5bd6846d156f89e7,
                    0x03194a8dfae962fc,
                    0x5f91978ee21eebd6,
                    0x367f716277941b73,
                    0x00360904333abe7d,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xd417fe3e2c59d6d5,
                    0x8053b8f720855337,
                    0x8eaf9ce95a974105,
                    0x2dc8b5c271ed58da,
                    0xbf03fcfdf699c2fc,
                    0x019886ad57555f69,
                ])),
                Felt::new(BigInteger384([
                    0xfe162c2bb93dc50c,
                    0x1914addfba44f06e,
                    0x6e603188e8934b6f,
                    0x06009029985aef2a,
                    0x5b74665768d1eeae,
                    0x011acbd4541966a8,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xbe8a6a438ea08820,
                    0xc1ea2a0f6eebb1f1,
                    0x0f230d6c9c839d24,
                    0xd98b26b6d9684a95,
                    0xcab7a555c6d36d19,
                    0x00b8171d88dbca9f,
                ])),
                Felt::new(BigInteger384([
                    0x8351622548ba88d6,
                    0x77fa4e197581e6b8,
                    0x5fb29ed66b95d1ec,
                    0xea79999ba498623c,
                    0xadf3062dc4d55250,
                    0x00566ce733a1bb0d,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x3a6db3651551af0f,
                    0x0c04920c65e84746,
                    0x4c5ddb50e67ed2cd,
                    0x54eb09a199d54809,
                    0x409c276df094bafc,
                    0x00f085ba8db31b32,
                ])),
                Felt::new(BigInteger384([
                    0xb0bc7ce0bfb72b81,
                    0xd36d17381d585a99,
                    0x27ba1dcaa0531f03,
                    0x3969dfd8b4e595ed,
                    0xd6e31a723da7b93c,
                    0x000a05a436f9c359,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x19457d6dfbd9a1ef,
                    0x4dcf9f8460509ddf,
                    0x242674bf2ba643e3,
                    0xc577015a9b5f1462,
                    0x7d758756c01e22b9,
                    0x015f82da646431b0,
                ])),
                Felt::new(BigInteger384([
                    0xb9cc0de5f7b82c3d,
                    0x35f3d956e5b05c28,
                    0xc3e02c22f51ba3cb,
                    0x87a4cb2c259de387,
                    0xc9b826871beda3b1,
                    0x001f04346e41ef08,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xd00d2444561199cb,
                    0x12bc0e46ddef2672,
                    0xff9912f9ee9e2577,
                    0xda4f7c1a2b99409d,
                    0xaae21420f51d1d10,
                    0x01189e3e4af53e94,
                ])),
                Felt::new(BigInteger384([
                    0x0d386bdfa20f52b7,
                    0x6d23cea4ee94bdb1,
                    0xbf67afef3e598d97,
                    0xb1a63679129efb5a,
                    0x7112dcac5a0da2ef,
                    0x008c4dfce7c228dc,
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
