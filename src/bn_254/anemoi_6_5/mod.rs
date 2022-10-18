//! Implementation of the Anemoi permutation

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

/// Function state is set to 6 field elements or 192 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 6;
/// 5 elements of the state are reserved for rate.
pub const RATE_WIDTH: usize = 5;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 3;

/// One element (32-bytes) is returned as digest.
pub const DIGEST_SIZE: usize = 1;

/// The number of rounds is set to 10 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 10;

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
    apply_mds_internal(&mut state[..NUM_COLUMNS]);
    state[NUM_COLUMNS..].rotate_left(1);
    apply_mds_internal(&mut state[NUM_COLUMNS..]);
}

#[inline(always)]
fn apply_mds_internal(state: &mut [Felt]) {
    let tmp = state[0] + mul_by_generator(&state[2]);
    state[2] += state[1];
    state[2] += mul_by_generator(&state[0]);

    state[0] = tmp + state[2];
    state[1] += tmp;
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
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let mut input = [
            [Felt::zero(); 6],
            [Felt::one(); 6],
            [
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            [
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0xadaac62d755beaa3,
                    0xf3bf24f027d8342a,
                    0xa5a5315285cbbb17,
                    0x2fad177cd90b5d97,
                ])),
                Felt::new(BigInteger256([
                    0xdd25037bb6af8ae0,
                    0x63f88a19f6d142cc,
                    0xbebf2de05adba5bb,
                    0x2809ae3ee64c45e4,
                ])),
                Felt::new(BigInteger256([
                    0x7a613b861b2fa79f,
                    0x5725a9399426ef04,
                    0x0a593f8b283ab8e3,
                    0x07e0b55025deddb2,
                ])),
                Felt::new(BigInteger256([
                    0x8228e12f684a57cd,
                    0xdcad4c550462756e,
                    0x9d9896dff87c884e,
                    0x10660e225c038884,
                ])),
                Felt::new(BigInteger256([
                    0x2a3255129b1d4f64,
                    0x51116875f9537ff6,
                    0x91eda75fff59f6ff,
                    0x094f05f17cde6d26,
                ])),
                Felt::new(BigInteger256([
                    0xf390ab284917942f,
                    0xd5d911c78fd77f0e,
                    0x311918c919bd3897,
                    0x21b651dd23fdb33d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7672577aeb95b2af,
                    0x56c2ed28e9cc9539,
                    0xee801d4de0377373,
                    0x0b92ee7b5d2890d9,
                ])),
                Felt::new(BigInteger256([
                    0xc917d55bff5c431e,
                    0xfd102971e78252b2,
                    0xa02ed920f18731c9,
                    0x24b2651e22109de8,
                ])),
                Felt::new(BigInteger256([
                    0x13597f51b9d4313b,
                    0x82d7f2bdb4988512,
                    0x16640a577fb5650e,
                    0x10b9a10b4ffe8c90,
                ])),
                Felt::new(BigInteger256([
                    0xfd0bfde397fb5dbe,
                    0x9460081d3121519a,
                    0xbea7b1ffefdd7f4d,
                    0x18ffac398c7e7836,
                ])),
                Felt::new(BigInteger256([
                    0x9a23e2106cb41dad,
                    0xf0359736247417fb,
                    0xe65cd5394dbda4ad,
                    0x25ed80c8fbba2ced,
                ])),
                Felt::new(BigInteger256([
                    0xbfc251fd0f440eca,
                    0x5a6fe3e27f1e2895,
                    0x5ad63f78759c389b,
                    0x10f2c17a912ff1bd,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbd72187476a193ec,
                    0xe7cc8914eb472b5b,
                    0x9ba84f8502108fad,
                    0x1c992b1ca52c568c,
                ])),
                Felt::new(BigInteger256([
                    0x25efd3c6a4ec93f6,
                    0xa927f9f60c091814,
                    0xae663d43325d0f30,
                    0x168a7fa952452a05,
                ])),
                Felt::new(BigInteger256([
                    0x0d773bec8026dc06,
                    0x1a21509395faf87f,
                    0xae601adfd3b41418,
                    0x1c98840949c61d31,
                ])),
                Felt::new(BigInteger256([
                    0x7ce6592ef69604e7,
                    0x04a2cba43ba97722,
                    0x57bfe1f5e092f25a,
                    0x0c3f43f1339245ba,
                ])),
                Felt::new(BigInteger256([
                    0xfccab98d2f18386b,
                    0xb097d1b4be056fd6,
                    0x00a12cb6231faf45,
                    0x2ba3470b5841a15c,
                ])),
                Felt::new(BigInteger256([
                    0x5e616dc5bcae376d,
                    0x140cd6cd86504aa6,
                    0x2e9e986dfa2125dd,
                    0x09c45e0d158acca9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1bc7265a2b7c1eb0,
                    0x264528eef6991bfc,
                    0xedd3af46a2da58f8,
                    0x068e9059e39790fe,
                ])),
                Felt::new(BigInteger256([
                    0xefa2c9138cd4b66d,
                    0xefa37c5ed9059d9f,
                    0x6057e7807b3131fa,
                    0x2f5c330bcc079a38,
                ])),
                Felt::new(BigInteger256([
                    0xe8933e783adaecfa,
                    0xe200912c92d33ba6,
                    0xb1a87ae519b69940,
                    0x1328331d9c4ebc5f,
                ])),
                Felt::new(BigInteger256([
                    0x540f03b580a0486c,
                    0xe4393973755979a5,
                    0xd33507489cfe79a5,
                    0x13810b020c4b0136,
                ])),
                Felt::new(BigInteger256([
                    0xbf1684efa98ee178,
                    0x2be7c49708394da6,
                    0xdb942d2d191e0334,
                    0x0fd4c0198d4e2bcd,
                ])),
                Felt::new(BigInteger256([
                    0xc0c7d54c3c0a8c80,
                    0xc08b58d8eef4f488,
                    0x0818a4c6c259d7a6,
                    0x029b6938d50ab944,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf4e3ebd3bab18733,
                    0xd925f557bfe83ffc,
                    0x20bda0eb67a0f72e,
                    0x0fefdb45ac664d56,
                ])),
                Felt::new(BigInteger256([
                    0x9b789eb5afb81d68,
                    0xac0aa69286df94af,
                    0x7e410e77fae8a3ae,
                    0x158540358c796915,
                ])),
                Felt::new(BigInteger256([
                    0x3ffb9ace13f5dfd7,
                    0xee87cc7c28af6f3d,
                    0x3cc7583d8c555243,
                    0x0539101372a27e14,
                ])),
                Felt::new(BigInteger256([
                    0x3d4a470e25fe07ad,
                    0x8ebb94446be17a64,
                    0x6a121c6d57e00304,
                    0x0b464c658ca4a204,
                ])),
                Felt::new(BigInteger256([
                    0xd9de16f7106d8eba,
                    0x033c38c1b494e2fd,
                    0xda3652fa0d7c183c,
                    0x303b08700ddc7d0a,
                ])),
                Felt::new(BigInteger256([
                    0xa5bdb60129ba1803,
                    0xf631f87c6e08af59,
                    0x01485dfe5acd35a3,
                    0x1274340a36b69fcb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3670481b241e058a,
                    0x6a0da17611ab6ad7,
                    0xfb11811dfc9b6a23,
                    0x23ac65d0def8fef7,
                ])),
                Felt::new(BigInteger256([
                    0x7cf8cf5a66e26a4a,
                    0xb8f6397f6c782f54,
                    0x6d027ac3fef944a8,
                    0x2fd34e8caa6b4d25,
                ])),
                Felt::new(BigInteger256([
                    0xda481fef17131f31,
                    0x7657304a655275de,
                    0x890652b2270773ab,
                    0x2951db049b5b258f,
                ])),
                Felt::new(BigInteger256([
                    0xefac8c0ef0f493b0,
                    0x62fc425c4c81ae18,
                    0xc928822e458d9434,
                    0x2ef354c4ea6e53e7,
                ])),
                Felt::new(BigInteger256([
                    0x23bf3d95b08931ea,
                    0x25ac701b84fbab36,
                    0xea40c6274a2ef7b1,
                    0x2b992c3f76ad1034,
                ])),
                Felt::new(BigInteger256([
                    0x2b86e853ce1e3ed0,
                    0x70db9e36f776eed3,
                    0xbbc25a1e56d18eca,
                    0x02528b65fe023a6f,
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
                Felt::new(BigInteger256([
                    0xafd49a8c34aeae4c,
                    0xe0a8c73e1f684743,
                    0xb4ea4db753538a2d,
                    0x14cf9766d3bdd51d,
                ])),
                Felt::zero(),
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
                Felt::new(BigInteger256([
                    0x68c3488912edefaa,
                    0x8d087f6872aabf4f,
                    0x51e1a24709081231,
                    0x2259d6b14729c0fa,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd18c1be9d9a111da,
                    0x38b6ba0c691bed31,
                    0x792b025d4d4d67b9,
                    0x0b2e72d2f8d83b0e,
                ])),
                Felt::new(BigInteger256([
                    0xb03b61b592fbbcb4,
                    0xa6977bd6e945cbf4,
                    0x3a2d8300d500c4e3,
                    0x26be7d5f3b5aef0c,
                ])),
                Felt::new(BigInteger256([
                    0x2f94c5ccd2fea727,
                    0x318046db04affe11,
                    0xcad7cd8825c08251,
                    0x108a45d0e901b29f,
                ])),
                Felt::new(BigInteger256([
                    0x314136ac133a47ca,
                    0xefe4313c0a43fc71,
                    0x19a419a109842a86,
                    0x03ee34972c41d31b,
                ])),
                Felt::new(BigInteger256([
                    0xcd85dbc32d142948,
                    0x3ec222c82d2dbd9e,
                    0x1c21573034d26a63,
                    0x07294d8b6eaa4e75,
                ])),
                Felt::new(BigInteger256([
                    0x2e48665585b7b207,
                    0x182a017ada16964a,
                    0xec83122e463f50ec,
                    0x0ec67348197f62fc,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x85b41daf66b98e30,
                    0xbf651397c5ecc21e,
                    0xc160daec4350efaf,
                    0x0dcbb6418613cb2e,
                ])),
                Felt::new(BigInteger256([
                    0x0c7e77bc35339246,
                    0xc2b425535580ca5c,
                    0xcc6415abbded4e26,
                    0x26138e48acd6819a,
                ])),
                Felt::new(BigInteger256([
                    0xb86a5db8d5f94bec,
                    0x02beeffeadd1edfa,
                    0x43bb5a38863cd4fe,
                    0x20d41948f6a8787b,
                ])),
                Felt::new(BigInteger256([
                    0xd59bedfd5094cce3,
                    0x38218fdd466cc050,
                    0x5e480478c5b227c6,
                    0x27bb5f64a78465c2,
                ])),
                Felt::new(BigInteger256([
                    0xb61ab7e7170c1560,
                    0xf3292c412156b00e,
                    0x803c80ec199513e6,
                    0x25f910937f6c2709,
                ])),
                Felt::new(BigInteger256([
                    0xf47c1de40e6ba895,
                    0xe4c4da7d76b9dc7e,
                    0x89c94ba1444bf73d,
                    0x2fb2f962893aad53,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x292a229a5e8f6add,
                    0x990814bcfa966bf1,
                    0x38107fb9fd88bf7d,
                    0x0e28764261c0e81d,
                ])),
                Felt::new(BigInteger256([
                    0x071e0b879a7d19cb,
                    0xb1ce5b4ff43aaa80,
                    0x014129057672ae89,
                    0x0dade80591dde2d4,
                ])),
                Felt::new(BigInteger256([
                    0x0ba7c983fb0b90eb,
                    0x9998f1a8f9e8fffb,
                    0x9a02086cd186a515,
                    0x09c4a9c18bfcf656,
                ])),
                Felt::new(BigInteger256([
                    0xcba3a2274fe97756,
                    0x8a78bccb5f91e71e,
                    0xf6ab96d80d0d7e78,
                    0x0b68ca7c54c0d1b4,
                ])),
                Felt::new(BigInteger256([
                    0x9171b7e7dbe59653,
                    0x8a3644651cdad200,
                    0x88e02a9786664527,
                    0x176fe42784af24f2,
                ])),
                Felt::new(BigInteger256([
                    0xa7bf3ed6648c2f4d,
                    0xa0544de077456966,
                    0x710aafff5cf74b8f,
                    0x2b62c7386794fe58,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdfe3335bd9502953,
                    0x3bea8dc07d39974a,
                    0xc10ecd78d6e86ea3,
                    0x104458ba074c5e31,
                ])),
                Felt::new(BigInteger256([
                    0x1837468c3e1c3463,
                    0x5fc352537beabeb0,
                    0x8453c8a8a3f3df45,
                    0x0acd3417f68623d3,
                ])),
                Felt::new(BigInteger256([
                    0x0df070b65afdb778,
                    0xf3395e4d4b38cbba,
                    0x4379393e8ea9e6a6,
                    0x1910da617f155bbd,
                ])),
                Felt::new(BigInteger256([
                    0xbf071299a89ca6fd,
                    0x8a6b101642d97521,
                    0x20f6b04a4f068110,
                    0x0b3dfdb2677b32e5,
                ])),
                Felt::new(BigInteger256([
                    0xd3a7f80380b22197,
                    0x343af9cc8efb71f1,
                    0x282f12b735cc70f9,
                    0x0c83ec7d5d150b0d,
                ])),
                Felt::new(BigInteger256([
                    0xef3f47f52b12bd0a,
                    0xe2df51b925506030,
                    0x32dcbb879a58101a,
                    0x0bf69099f1568bd7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x022d01b796dd2a46,
                    0xcfa1b0efc544be14,
                    0x872871d541e1afe0,
                    0x094bd41e5d4e22dd,
                ])),
                Felt::new(BigInteger256([
                    0x29870cde9795b056,
                    0x3cb87b3369ca9c2c,
                    0x70611db3e0427c61,
                    0x197eea1e62468534,
                ])),
                Felt::new(BigInteger256([
                    0x94a981f7dab2539d,
                    0xba91d23cc41987af,
                    0x96a86121535bb342,
                    0x1d204a161b8c90d6,
                ])),
                Felt::new(BigInteger256([
                    0x638aa1cdfea8d50f,
                    0xd32aa52318518516,
                    0x9d32f6f4a5c4e925,
                    0x11184a107911b7a4,
                ])),
                Felt::new(BigInteger256([
                    0x140db2051a16f97c,
                    0x81ac83f2f322188b,
                    0x93af968418e91b2b,
                    0x2c4688d70702209c,
                ])),
                Felt::new(BigInteger256([
                    0x09f630bb997a283a,
                    0xafec463cbbc3b17d,
                    0x809f8ee7366d5840,
                    0x19e6828a3ae7804f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf2424319cec39571,
                    0xc5499b391aba49d7,
                    0xfda922405bb058d7,
                    0x07eeabf08bb4f895,
                ])),
                Felt::new(BigInteger256([
                    0x774d1dc480fe102d,
                    0xcb5bd82398ac8965,
                    0xd0422ef74f4ea532,
                    0x2a34be65b7bdb391,
                ])),
                Felt::new(BigInteger256([
                    0x5b60eba9194e8c4a,
                    0x4450f3cc27e31bfe,
                    0x9c45ef4006f7a0c1,
                    0x1d1470377c1388e4,
                ])),
                Felt::new(BigInteger256([
                    0x85376b2b45d4fcf1,
                    0xdd9033cbef1efe01,
                    0x570bc53bacdd9ded,
                    0x1735d834dca33b13,
                ])),
                Felt::new(BigInteger256([
                    0xaac5e5b380df9a94,
                    0x58b709c639e18e50,
                    0x715e34cba7b0006a,
                    0x0ed6d79a4e01ccd8,
                ])),
                Felt::new(BigInteger256([
                    0xa953d7a7f6e74474,
                    0xec66b22f7f2a6ad5,
                    0x3ee0c21a18e86da2,
                    0x24d4f733699654b3,
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
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let mut input = [
            [Felt::zero(); 6],
            [Felt::one(); 6],
            [
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            [
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0xe6d7461b5d414e16,
                    0x1c71f0a9b7ac587b,
                    0xebf03757e1421e70,
                    0x0ed372433b14ffda,
                ])),
                Felt::new(BigInteger256([
                    0xdcf7914d05d62563,
                    0x735928045e1a8d4b,
                    0xfb0ba36a0ff1cccf,
                    0x2f11d6c92ead83c5,
                ])),
                Felt::new(BigInteger256([
                    0xb4289a4881e8cb67,
                    0x9e87ea6e46028aee,
                    0xdd7d91ce2bb1b14c,
                    0x2bf4de33629ef0ed,
                ])),
                Felt::new(BigInteger256([
                    0xb6b4a9465cb460e3,
                    0xd3865721310405fa,
                    0xc953ef5b1b163127,
                    0x21901f19b6068afb,
                ])),
                Felt::new(BigInteger256([
                    0x567208cb2d32bf11,
                    0x8751ceb8be1df8aa,
                    0x01d64f40e6105ff0,
                    0x05df9fbc229f745f,
                ])),
                Felt::new(BigInteger256([
                    0x846e52eb8eb95ae4,
                    0xfc545beaad396cf4,
                    0xddfe4089be12cd1b,
                    0x05691ea957d23ba5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x315cda9bce031f7c,
                    0xce916a58eed01029,
                    0x8e1a39e44834e899,
                    0x254c3a140a995eb8,
                ])),
                Felt::new(BigInteger256([
                    0xbff5417a0e297afe,
                    0x716ba91ef4e93960,
                    0x4566cdb2eef6c2fe,
                    0x1e5a03178c327c0a,
                ])),
                Felt::new(BigInteger256([
                    0xb145b562be58b936,
                    0xb855cb79791a1eed,
                    0x0a291064df7bfea1,
                    0x082c2e7b0fde9bb9,
                ])),
                Felt::new(BigInteger256([
                    0x5b8a557f4da1f4e2,
                    0x403d53367af628dd,
                    0x1543a8d7672af643,
                    0x2a5a8b7edb3855a1,
                ])),
                Felt::new(BigInteger256([
                    0xd5af4a07851b9d93,
                    0x9472035f51af69b5,
                    0xe440efd22b7a5660,
                    0x03120a6c45a229a1,
                ])),
                Felt::new(BigInteger256([
                    0xef4599a6cbfa9edb,
                    0xadd51abaab708008,
                    0x2aef3836e59d7b64,
                    0x08f1fab0f7ea30ac,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb4beec74dcd8d85d,
                    0x4b9226fef7ac9261,
                    0x74cffdfe60fe74bf,
                    0x1756766ed6443d2a,
                ])),
                Felt::new(BigInteger256([
                    0x3aa12c8c26fb6b7a,
                    0x55ffdfc05f713e83,
                    0xd82430ce92dad88d,
                    0x0a72169bb09b1fff,
                ])),
                Felt::new(BigInteger256([
                    0xebed59921b4f6b7d,
                    0x1b3e8f866dba3dfd,
                    0xcfe5195109f20851,
                    0x2c88a1f011daf726,
                ])),
                Felt::new(BigInteger256([
                    0xc86f4b78c250888b,
                    0x61a1dc4589a30a8c,
                    0xfb84a68eaca58471,
                    0x0ce4e30247d03709,
                ])),
                Felt::new(BigInteger256([
                    0x49697a0f1775dccb,
                    0xebc60520368d2f6e,
                    0x466b7450b7656ee3,
                    0x09d99c809d0f056b,
                ])),
                Felt::new(BigInteger256([
                    0x22856e12acc10f6c,
                    0x415918df130305d1,
                    0xad4b7fcb5577a18a,
                    0x26acae0a9f5aa50a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9ae31e0cb0e956f8,
                    0xee71d8801fdd49c4,
                    0x36e7ad0904b9de0c,
                    0x061d0383620c7572,
                ])),
                Felt::new(BigInteger256([
                    0x661325ef4baebbc9,
                    0x09616c6c4c7e48b2,
                    0xbf969958a604fe1f,
                    0x1519a95cbbfecd4e,
                ])),
                Felt::new(BigInteger256([
                    0xc7463bf81d14c3da,
                    0xfdc6ee359512ab60,
                    0x85e0021c32d59031,
                    0x2c597f5d396a3d04,
                ])),
                Felt::new(BigInteger256([
                    0x46afd968471f6522,
                    0xd09820a69513de9f,
                    0xb90a11e089ffd532,
                    0x020857f22967e1ae,
                ])),
                Felt::new(BigInteger256([
                    0x6c90af0d667ed532,
                    0x4783d9f64fe50ffa,
                    0x62e0263d7905bb46,
                    0x1b91f78ebb626848,
                ])),
                Felt::new(BigInteger256([
                    0x128a30ba73973cb6,
                    0xd04f3e50a6b4a235,
                    0x64e07d06db523056,
                    0x1dca749a6788e866,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfecbdc9691273beb,
                    0x30a3eefc44d1211d,
                    0xd337725f59cb51cf,
                    0x27eeef757bd4bcd5,
                ])),
                Felt::new(BigInteger256([
                    0x546b69e3d7f60d20,
                    0x730c7aed837a327d,
                    0xcdc0cc707b25efad,
                    0x115bbc722081fd3e,
                ])),
                Felt::new(BigInteger256([
                    0xf619d30d7b4f18e2,
                    0x4c6cb446ef5fb08a,
                    0x31c285246870f70e,
                    0x144afa85d8e50710,
                ])),
                Felt::new(BigInteger256([
                    0xa0c30c3ff1ebf135,
                    0x86411de3b093c159,
                    0xde607e5b32ef0450,
                    0x1f2bc5eb45999a2f,
                ])),
                Felt::new(BigInteger256([
                    0x848128bf48410b2e,
                    0x67cba9098be349f1,
                    0x22180fb0be095ee7,
                    0x15417ec0dc55441c,
                ])),
                Felt::new(BigInteger256([
                    0x8c59e28e1c03d5be,
                    0x877b440d36a460dc,
                    0x2d8589335dfc9eb8,
                    0x18a47c10f5c9aac6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd9c69fcfbce9599b,
                    0x7941e79c10b6c246,
                    0xbe30dc525bb8f09a,
                    0x1f172ccecd7b64af,
                ])),
                Felt::new(BigInteger256([
                    0xcd2110ceb971092c,
                    0xcf2b2cf35df5ff8c,
                    0x564ca84acec0ce67,
                    0x0c47eaa789ea1235,
                ])),
                Felt::new(BigInteger256([
                    0x6729e34f366e9ce1,
                    0x14b22200ecf39d10,
                    0x2951b746d82029f7,
                    0x003e0d3dbf337edf,
                ])),
                Felt::new(BigInteger256([
                    0x3cb7934560df07a9,
                    0x662c5423322f663d,
                    0x3648afe7836d5330,
                    0x2f0baf8c348f8660,
                ])),
                Felt::new(BigInteger256([
                    0xe75a766241b7e2ce,
                    0xae220267d494488c,
                    0xd8d33e8b78c9ed95,
                    0x222e83e2489b3391,
                ])),
                Felt::new(BigInteger256([
                    0x6073e58a09f2a4d7,
                    0xbd28c2d75e9fa510,
                    0xbf0256d61b7ec275,
                    0x2429001a0981227d,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(); 6],
            [
                Felt::new(BigInteger256([
                    0xf60647ce410d7ff7,
                    0x2f3d6f4dd31bd011,
                    0x2943337e3940c6d1,
                    0x1d9598e8a7e39857,
                ])),
                Felt::new(BigInteger256([
                    0xe4b1c5ae034e46ca,
                    0x9cdb2d3b64716da7,
                    0x47d8eb76d8dd067e,
                    0x15d0085520f5bbc3,
                ])),
                Felt::new(BigInteger256([
                    0xe4b1c5ae034e46ca,
                    0x9cdb2d3b64716da7,
                    0x47d8eb76d8dd067e,
                    0x15d0085520f5bbc3,
                ])),
                Felt::new(BigInteger256([
                    0xf60647ce410d7ff7,
                    0x2f3d6f4dd31bd011,
                    0x2943337e3940c6d1,
                    0x1d9598e8a7e39857,
                ])),
                Felt::new(BigInteger256([
                    0xe4b1c5ae034e46ca,
                    0x9cdb2d3b64716da7,
                    0x47d8eb76d8dd067e,
                    0x15d0085520f5bbc3,
                ])),
                Felt::new(BigInteger256([
                    0xe4b1c5ae034e46ca,
                    0x9cdb2d3b64716da7,
                    0x47d8eb76d8dd067e,
                    0x15d0085520f5bbc3,
                ])),
            ],
            [
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::new(BigInteger256([
                    0xf60647ce410d7ff7,
                    0x2f3d6f4dd31bd011,
                    0x2943337e3940c6d1,
                    0x1d9598e8a7e39857,
                ])),
                Felt::new(BigInteger256([
                    0xe4b1c5ae034e46ca,
                    0x9cdb2d3b64716da7,
                    0x47d8eb76d8dd067e,
                    0x15d0085520f5bbc3,
                ])),
                Felt::new(BigInteger256([
                    0xe4b1c5ae034e46ca,
                    0x9cdb2d3b64716da7,
                    0x47d8eb76d8dd067e,
                    0x15d0085520f5bbc3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf60647ce410d7ff7,
                    0x2f3d6f4dd31bd011,
                    0x2943337e3940c6d1,
                    0x1d9598e8a7e39857,
                ])),
                Felt::new(BigInteger256([
                    0xe4b1c5ae034e46ca,
                    0x9cdb2d3b64716da7,
                    0x47d8eb76d8dd067e,
                    0x15d0085520f5bbc3,
                ])),
                Felt::new(BigInteger256([
                    0xe4b1c5ae034e46ca,
                    0x9cdb2d3b64716da7,
                    0x47d8eb76d8dd067e,
                    0x15d0085520f5bbc3,
                ])),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0x1c54566a480d98f4,
                    0x69b97f8d4a9d2634,
                    0x87316b71bc3a51ee,
                    0x283d90653f852618,
                ])),
                Felt::new(BigInteger256([
                    0xefc675e686dde092,
                    0x0d5d2db346075c5e,
                    0x9e3379526e439daf,
                    0x0032a9db0cd8d5c3,
                ])),
                Felt::new(BigInteger256([
                    0xcd64e5b9ee88e07e,
                    0x38340f4cfa3e8c93,
                    0x2bb94fd2dc6728b1,
                    0x26b86ee0802833f1,
                ])),
                Felt::new(BigInteger256([
                    0x04a776ed2cdee2df,
                    0xa130b39e306c07e1,
                    0xe1b669d63e290864,
                    0x11fb2ea816d55893,
                ])),
                Felt::new(BigInteger256([
                    0x86bd3f5c210f4210,
                    0xcf365ae42d7fe274,
                    0xcb2fd26ef2630fc8,
                    0x0f307eccda2210a4,
                ])),
                Felt::new(BigInteger256([
                    0x02588a7c9a88fbb3,
                    0xce4eb4a4b0259261,
                    0xf484d7f109d8c5b7,
                    0x0833ce8494858394,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x59fd5118dda4e8aa,
                    0x2f02d622f2cacb87,
                    0xc532dffd87b4fe76,
                    0x12aa6b88714be529,
                ])),
                Felt::new(BigInteger256([
                    0xc902b0273eb9c8d5,
                    0xd17d0b52e695dbc5,
                    0x39abf30f541e4f1f,
                    0x2bc67a29e5360dc4,
                ])),
                Felt::new(BigInteger256([
                    0x50efe26bad149ad3,
                    0xcef173ef011e2922,
                    0xd0edbaa1228d7254,
                    0x053df4761848536f,
                ])),
                Felt::new(BigInteger256([
                    0xffca737d8d79f0da,
                    0x3a0e355da4b16aad,
                    0xe810c9b9abaea4db,
                    0x2d776704d7bf4d3a,
                ])),
                Felt::new(BigInteger256([
                    0x5f52cbfe89022086,
                    0xd3fc429a9d1ecf3c,
                    0xde5a9722439603d3,
                    0x2a4b0ab40cd21add,
                ])),
                Felt::new(BigInteger256([
                    0x8fbd4125d0726f2f,
                    0x13e70d7db3031b7a,
                    0x34a56ace4db61c6c,
                    0x0c1e5701c2d76309,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x90af8835cd2b887f,
                    0xfbbba4feead38b3f,
                    0x5167317bb71612fa,
                    0x27f8efd8eb1fd074,
                ])),
                Felt::new(BigInteger256([
                    0xfec68172cc4b8e79,
                    0x2cc9759e66f72b36,
                    0x93b2a99c8d2b5d27,
                    0x1635878218db6221,
                ])),
                Felt::new(BigInteger256([
                    0xcc8a334f27db6580,
                    0x24f20f20e34d9e8b,
                    0x95d8b8adbcc58e61,
                    0x1c357ef282df8e52,
                ])),
                Felt::new(BigInteger256([
                    0xf1a76c0462e0aa36,
                    0x47f5c95342e058a2,
                    0x446b5fdbe2a0be24,
                    0x20de0f307074568c,
                ])),
                Felt::new(BigInteger256([
                    0x891c3e7532ab8891,
                    0xba83483e7e078a58,
                    0x2df4a211914c4564,
                    0x26d0a51f32a8af6a,
                ])),
                Felt::new(BigInteger256([
                    0x8b109ba1dcf63111,
                    0xcecb99f3d7dbd41b,
                    0xc3c23d95a6cc1a49,
                    0x20ba181bdd264c2c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfe365da721b331f5,
                    0x5c3edcfd7e76f312,
                    0xd1743f137e3d55a4,
                    0x1d627b13a5131682,
                ])),
                Felt::new(BigInteger256([
                    0xa267539fca5f667a,
                    0x2aa3cfd8f23e34f1,
                    0x5f2d7b92bebb83a9,
                    0x0f163f9f26b51951,
                ])),
                Felt::new(BigInteger256([
                    0xc1e22ff6a3028744,
                    0x3afc7990d8b706d3,
                    0x31dd5cd96586d01a,
                    0x2365e4d13a5cca80,
                ])),
                Felt::new(BigInteger256([
                    0x2b2aae4ca0992e31,
                    0x6a3ae9100142fcf4,
                    0xab988c5b62e46922,
                    0x0306c745571d2fc5,
                ])),
                Felt::new(BigInteger256([
                    0x1709dfe9d6f74407,
                    0xf21a0fa94d638380,
                    0x3a8e932f70d612d7,
                    0x0f11258cbdf15591,
                ])),
                Felt::new(BigInteger256([
                    0x26aaff1d3d3926e0,
                    0x487017b75a941ba9,
                    0xd5ea7632cd6086a1,
                    0x11c0165300b4c29a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfb5f6c01cf5e6df1,
                    0x71c7f3234a04845e,
                    0x48174deefc90594f,
                    0x104ddc210d70ec06,
                ])),
                Felt::new(BigInteger256([
                    0xbd43a7752a109923,
                    0x59f3b19bc586d021,
                    0xc59f42d00b4175ec,
                    0x1562fe9364a28ef1,
                ])),
                Felt::new(BigInteger256([
                    0x92872e707d43e1ee,
                    0x8ae0bc7507f7e6ba,
                    0x5038d78f6c74d310,
                    0x0c4699ffc9505a53,
                ])),
                Felt::new(BigInteger256([
                    0x30e8862fa2c3d22e,
                    0xe1a8b57c86b963d4,
                    0x4e26aa891bd8ca21,
                    0x28c854f5f8bea34f,
                ])),
                Felt::new(BigInteger256([
                    0x7ae317df890eb9fd,
                    0x5307719f035f59c0,
                    0x7a1e8888b1d059d6,
                    0x2aa0afade0887d1e,
                ])),
                Felt::new(BigInteger256([
                    0x425f50de35b8edef,
                    0x161c87eab9fe6af0,
                    0x018dab33c8050f04,
                    0x16cc21590dffd0f7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x58a2051cd5d6e88e,
                    0xd7f87e4483bbe7cf,
                    0x83b66b429b2287f2,
                    0x28d435f3fa42601d,
                ])),
                Felt::new(BigInteger256([
                    0xdc655a8c19a6396a,
                    0x86837a9235879904,
                    0x9072aa71b2da3ce7,
                    0x2c193f2f94fff382,
                ])),
                Felt::new(BigInteger256([
                    0x495dbb5f75a1b850,
                    0x20a030a5ac2a4e57,
                    0x4990691bb7091973,
                    0x0902e16bef2c7ed0,
                ])),
                Felt::new(BigInteger256([
                    0x4bd83788a8e361c2,
                    0xe9d833099e91d65c,
                    0xf14028a481d25afe,
                    0x1653a8afd5d0a921,
                ])),
                Felt::new(BigInteger256([
                    0x0d72e5610c53a984,
                    0x3fca176327faf620,
                    0x596e8e3e188b4826,
                    0x11e958d56b046889,
                ])),
                Felt::new(BigInteger256([
                    0x9ed937b1a6825d15,
                    0x6736de7dd536854c,
                    0x56d3f13c84c5d54e,
                    0x28934ff4744d6316,
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
