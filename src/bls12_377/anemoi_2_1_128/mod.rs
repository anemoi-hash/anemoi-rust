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

/// One element (48-bytes) is returned as digest.
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
