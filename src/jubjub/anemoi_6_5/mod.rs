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

/// Applies the Anemoi S-Box on the current
/// hash state elements.
#[inline(always)]
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

/// Applies matrix-vector multiplication of the current
/// hash state with the Anemoi MDS matrix.
#[inline(always)]
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
                    0x9c44201374afc048,
                    0x24fa89db81f545b5,
                    0x2ab19b18142046c3,
                    0x14253465c1a3b992,
                ])),
                Felt::new(BigInteger256([
                    0x2e04d7b7df4431a5,
                    0x64cb146dafd0e3c8,
                    0x8607ef7abfebe70c,
                    0x706d5325973aee55,
                ])),
                Felt::new(BigInteger256([
                    0x9234215d71b583cc,
                    0x4c956670626a223b,
                    0x1508fd87b86c65ec,
                    0x48caa582e5c6ba8f,
                ])),
                Felt::new(BigInteger256([
                    0x9fb052dc2df31cb3,
                    0x024aebb958622eb4,
                    0x2d3da60eac77edd3,
                    0x31e178b3223ca373,
                ])),
                Felt::new(BigInteger256([
                    0x052ad4daf8097eeb,
                    0xf949aea4fa1a7edb,
                    0x8d0632069e0b63a9,
                    0x1055bbb8c82e9597,
                ])),
                Felt::new(BigInteger256([
                    0x393dbfe2ad92eb1a,
                    0xb2136aa59de50a59,
                    0xa4fd7147cc885081,
                    0x0c7d82fdc99f8396,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5a8298027541f7ea,
                    0x93bc2c6b6852dc04,
                    0x4f6a528213408854,
                    0x4f6f0f2a927ea861,
                ])),
                Felt::new(BigInteger256([
                    0x321b19c3506e0462,
                    0x975a7062fc6206a1,
                    0x9adbfc0c35d3654b,
                    0x647869aebad276cb,
                ])),
                Felt::new(BigInteger256([
                    0xb2ad3e82870da62e,
                    0xc5f0c3fc699f5fbc,
                    0x154de7c687004900,
                    0x1befd68f9f774483,
                ])),
                Felt::new(BigInteger256([
                    0x7d51b0ad3bf74ec6,
                    0x1834d0c3078c92f0,
                    0xc81089e36f27f04c,
                    0x560619dcc3e9715f,
                ])),
                Felt::new(BigInteger256([
                    0xd01d4fd4f648738f,
                    0x3e43a523491c9e13,
                    0xd99af330ffe38b95,
                    0x0826173bebd30eb2,
                ])),
                Felt::new(BigInteger256([
                    0x257acc809e7c7c94,
                    0xfe123c4980d7a6aa,
                    0x3c29e93642ccc8bc,
                    0x350efefff568af9b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5c1fa6f22412c317,
                    0x5da7eef15737f1f2,
                    0x6720c1225a70ce5b,
                    0x3c459835bfb390d2,
                ])),
                Felt::new(BigInteger256([
                    0x2f560015570934f6,
                    0x15a989f50d9cd109,
                    0x8ef7631a827a3581,
                    0x5a0177efa4ab60a7,
                ])),
                Felt::new(BigInteger256([
                    0xeea647d8455d1203,
                    0x820fa480b4aa2ee3,
                    0x19e00e079d994f48,
                    0x72970468ee6d49c0,
                ])),
                Felt::new(BigInteger256([
                    0xcf5042cfc1a7fb35,
                    0x003685690f7750ca,
                    0x3eb4f7bf924abc60,
                    0x2fed11c551c9a63a,
                ])),
                Felt::new(BigInteger256([
                    0x18fcb206141cd726,
                    0x2c013e1b911c07ea,
                    0x42f86488b970ed68,
                    0x5ad247fbcfdd3c50,
                ])),
                Felt::new(BigInteger256([
                    0x9ba56529f68beb44,
                    0xcd05be8fb753f023,
                    0x0c53a607f8f21f33,
                    0x6157ba771219a87a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8eef0b4cc3bc6f91,
                    0x72dded875107c726,
                    0xa5a45c6e30180517,
                    0x20ab1338cd1c4772,
                ])),
                Felt::new(BigInteger256([
                    0x008547aa11765aa8,
                    0x642fc42a0e738343,
                    0x8b27590a305ba5d8,
                    0x182872fa31aed907,
                ])),
                Felt::new(BigInteger256([
                    0x2609a2a3983db5df,
                    0x672dde1cc16169a0,
                    0x596baa61af212dc1,
                    0x40634605ba39c037,
                ])),
                Felt::new(BigInteger256([
                    0x07f7d9fb28994bab,
                    0x583972a0c2361589,
                    0x0d140461e843a798,
                    0x1c52ebe365b48c10,
                ])),
                Felt::new(BigInteger256([
                    0xf52243a7ad7403b4,
                    0xe72da1d4b3bcbc15,
                    0xa36c362ae9e07052,
                    0x6246e0d7fa2a08e3,
                ])),
                Felt::new(BigInteger256([
                    0x8962786f5d34936d,
                    0x907f5ec92ae4081f,
                    0xe03ab932407c79d7,
                    0x1daa29d56d463349,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb7e4b98def6b376e,
                    0x0aa371a8b9e14aad,
                    0x83f4aa6433913ef2,
                    0x1b31a69802848feb,
                ])),
                Felt::new(BigInteger256([
                    0xc3c25c30507e797c,
                    0xcdba6d3dfe0434d8,
                    0xbecc385c43332aa7,
                    0x1c1a9ff5727e30dc,
                ])),
                Felt::new(BigInteger256([
                    0x6833dc345393b190,
                    0x971f0b3ade228bcd,
                    0x73e129fa4aee6d64,
                    0x2fe514dd5164e8f9,
                ])),
                Felt::new(BigInteger256([
                    0xff4dcbba3f82e7c7,
                    0x9a59bea52f3be21d,
                    0xda82e153dcc98c36,
                    0x0a1e413fb8864492,
                ])),
                Felt::new(BigInteger256([
                    0x414e7e24f100af12,
                    0x59aab7d3284e70ad,
                    0xac47790bea0b0e41,
                    0x43e4344ecc938e27,
                ])),
                Felt::new(BigInteger256([
                    0x46d08d1bd32f898d,
                    0x3c7398a75bd2e31e,
                    0xf69bd114c68fbbf7,
                    0x508a7df8dc3dba57,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc80386029d9ff842,
                    0xf145008f60b3adfc,
                    0x449ae7a12403c145,
                    0x597ef480239deb53,
                ])),
                Felt::new(BigInteger256([
                    0x66eb4c24cbe84f00,
                    0x3d91b4b226f6b3aa,
                    0x27d3de0bda94c4cf,
                    0x43f2408ce7bf7f7e,
                ])),
                Felt::new(BigInteger256([
                    0x4b3e91bd0af5489c,
                    0x05c14f00fbcb4d85,
                    0xef00137e2d818d09,
                    0x027438b82a7f5b1e,
                ])),
                Felt::new(BigInteger256([
                    0xe6653362e2a3e42f,
                    0x415349e8d04a1e20,
                    0x7e603d074ce6cea9,
                    0x6a729bb9e2327e12,
                ])),
                Felt::new(BigInteger256([
                    0xf9ee608737096e7b,
                    0xea722ad4089fa4de,
                    0xda4e4bdf64c41f44,
                    0x1b6b435736b82c4d,
                ])),
                Felt::new(BigInteger256([
                    0xbe06a628a3c67bf4,
                    0xedbe819b6d77f61f,
                    0x9c39729c9ffb9a41,
                    0x19d8da4cd6638e87,
                ])),
            ],
        ];

        let output = [
            [
                Felt::new(BigInteger256([
                    0xdb6db6dadb6db6dc,
                    0xe6b5824adb6cc6da,
                    0xf8b356e005810db9,
                    0x66d0f1e660ec4796,
                ])),
                Felt::new(BigInteger256([
                    0xdb6db6dadb6db6dc,
                    0xe6b5824adb6cc6da,
                    0xf8b356e005810db9,
                    0x66d0f1e660ec4796,
                ])),
                Felt::new(BigInteger256([
                    0xdb6db6dadb6db6dc,
                    0xe6b5824adb6cc6da,
                    0xf8b356e005810db9,
                    0x66d0f1e660ec4796,
                ])),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0xd99f4a8fe00e11bd,
                    0xe12a753d82b33b30,
                    0xba87bf86c47e6f0b,
                    0x58de312b5993b67f,
                ])),
                Felt::new(BigInteger256([
                    0xd99f4a8fe00e11bd,
                    0xe12a753d82b33b30,
                    0xba87bf86c47e6f0b,
                    0x58de312b5993b67f,
                ])),
                Felt::new(BigInteger256([
                    0xd99f4a8fe00e11bd,
                    0xe12a753d82b33b30,
                    0xba87bf86c47e6f0b,
                    0x58de312b5993b67f,
                ])),
                Felt::new(BigInteger256([
                    0xb4c1972e044567f4,
                    0x8e14aacbecb9fb59,
                    0x2c0f182612f81cd9,
                    0x34d6a5db27797543,
                ])),
                Felt::new(BigInteger256([
                    0xb4c1972e044567f4,
                    0x8e14aacbecb9fb59,
                    0x2c0f182612f81cd9,
                    0x34d6a5db27797543,
                ])),
                Felt::new(BigInteger256([
                    0xb4c1972e044567f4,
                    0x8e14aacbecb9fb59,
                    0x2c0f182612f81cd9,
                    0x34d6a5db27797543,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x34413b9512617cf9,
                    0x7e6bdaab3f8443c5,
                    0x752178862c9a6ab7,
                    0x3dde7b14d2951772,
                ])),
                Felt::new(BigInteger256([
                    0x34413b9512617cf9,
                    0x7e6bdaab3f8443c5,
                    0x752178862c9a6ab7,
                    0x3dde7b14d2951772,
                ])),
                Felt::new(BigInteger256([
                    0x34413b9512617cf9,
                    0x7e6bdaab3f8443c5,
                    0x752178862c9a6ab7,
                    0x3dde7b14d2951772,
                ])),
                Felt::new(BigInteger256([
                    0xbdf2fa0b931e7a04,
                    0x60f39c6eae7a9a3c,
                    0x34e97cf1bd62a467,
                    0x13b9e8a487cc4745,
                ])),
                Felt::new(BigInteger256([
                    0xbdf2fa0b931e7a04,
                    0x60f39c6eae7a9a3c,
                    0x34e97cf1bd62a467,
                    0x13b9e8a487cc4745,
                ])),
                Felt::new(BigInteger256([
                    0xbdf2fa0b931e7a04,
                    0x60f39c6eae7a9a3c,
                    0x34e97cf1bd62a467,
                    0x13b9e8a487cc4745,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdb6db6ecdb6db6ca,
                    0x035ffa14db8a4eec,
                    0x5ea2264f581fdd5a,
                    0x401b2e0d73d97883,
                ])),
                Felt::new(BigInteger256([
                    0xdb6db6ecdb6db6ca,
                    0x035ffa14db8a4eec,
                    0x5ea2264f581fdd5a,
                    0x401b2e0d73d97883,
                ])),
                Felt::new(BigInteger256([
                    0xdb6db6ecdb6db6ca,
                    0x035ffa14db8a4eec,
                    0x5ea2264f581fdd5a,
                    0x401b2e0d73d97883,
                ])),
                Felt::new(BigInteger256([
                    0xfffffffd00000003,
                    0xfb38ec08fffb13fc,
                    0x99ad88181ce5880f,
                    0x5bc8f5f97cd877d8,
                ])),
                Felt::new(BigInteger256([
                    0xfffffffd00000003,
                    0xfb38ec08fffb13fc,
                    0x99ad88181ce5880f,
                    0x5bc8f5f97cd877d8,
                ])),
                Felt::new(BigInteger256([
                    0xfffffffd00000003,
                    0xfb38ec08fffb13fc,
                    0x99ad88181ce5880f,
                    0x5bc8f5f97cd877d8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe7ce578101022ee5,
                    0xc0aff6e2c369eba7,
                    0xf30e354c91253d03,
                    0x340b1b2281e49aff,
                ])),
                Felt::new(BigInteger256([
                    0x9d72d05c910c80cb,
                    0x1005cff269238db2,
                    0xcbe8059d2a89be9a,
                    0x0c0442a5cf432680,
                ])),
                Felt::new(BigInteger256([
                    0x46f0655764d3e506,
                    0x1c461a1ff7db2ebe,
                    0xb24eefaf9e3405d5,
                    0x0445721944eefc26,
                ])),
                Felt::new(BigInteger256([
                    0xc746c43c5d7f0a3b,
                    0xb461857c3a7965e1,
                    0x98354d239f93584c,
                    0x0288f26519150099,
                ])),
                Felt::new(BigInteger256([
                    0xa17157075b14dd46,
                    0xc77407830fbf8744,
                    0x2e6ea31f06550791,
                    0x52efdcb5df1a04a3,
                ])),
                Felt::new(BigInteger256([
                    0x9c99c832616cbb2d,
                    0xec86b0c185b9b2c1,
                    0x756e5b564885cc50,
                    0x0a8ca370c721ba43,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x74f4940eba32d504,
                    0x09e5c9383f2e072e,
                    0x78df2ec91fa99359,
                    0x238dc9d2fc798a5a,
                ])),
                Felt::new(BigInteger256([
                    0xce25678cb33341d0,
                    0x3286042c78f629cd,
                    0x083e5f0e2ac4c4c6,
                    0x2c98cb7b956002c9,
                ])),
                Felt::new(BigInteger256([
                    0xd83c316e42772762,
                    0xae23daff13a3f2d3,
                    0xf3a5826d42c111fd,
                    0x3729fbe3cd9920f8,
                ])),
                Felt::new(BigInteger256([
                    0xe2967335e29330f1,
                    0x3d73402fcf3aa5d6,
                    0xa7fcc966b562decb,
                    0x1832fc36197038d8,
                ])),
                Felt::new(BigInteger256([
                    0x37be0385336052e9,
                    0xbbda1793267957f0,
                    0x7258fabf75e7f1a8,
                    0x558177742fbd370f,
                ])),
                Felt::new(BigInteger256([
                    0x62effdff3df7520c,
                    0x8f61107240ae06a0,
                    0x29e781afa6800c4a,
                    0x4653aab8c8353e9d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x84d76d26af2f67e6,
                    0xe567d66f3b17eb31,
                    0x17c6d3aab64a95e3,
                    0x627314a43a815020,
                ])),
                Felt::new(BigInteger256([
                    0x4cabe286fdd9727f,
                    0x2274c47a12225d74,
                    0xb538db8544a9877c,
                    0x26ab5b3dcbbba11d,
                ])),
                Felt::new(BigInteger256([
                    0xa96bf8e0e63520bc,
                    0x3a551a9e56d6e1c6,
                    0x659c05d54debf133,
                    0x1ce1eacc356ccc62,
                ])),
                Felt::new(BigInteger256([
                    0x43a6090e7a665fd9,
                    0xd9d599f2b7d67542,
                    0xdcd0bf38383393eb,
                    0x54b84413b236a7b8,
                ])),
                Felt::new(BigInteger256([
                    0x33c3f8a77e8d9801,
                    0xc80700c570ecb38b,
                    0xda7fa867fb38847c,
                    0x0108786e6352d10a,
                ])),
                Felt::new(BigInteger256([
                    0x1606427bc49da14c,
                    0x3eeae9d1f9c562dc,
                    0xe87e041cfa246b9f,
                    0x6291b9d1dd134d08,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6d5285c9ce050bda,
                    0x664437f207520e08,
                    0x0cb9a21599f27b79,
                    0x2e5b396da03d9802,
                ])),
                Felt::new(BigInteger256([
                    0x8c422622ff1884db,
                    0x205e744d4441f8f6,
                    0xeaceb036bc0ecfa1,
                    0x382abed90b1a35f8,
                ])),
                Felt::new(BigInteger256([
                    0xcf95ff521a3ea0de,
                    0x857b47cea3329a6e,
                    0x1bc44e9aeaa45be5,
                    0x3902072379a687b9,
                ])),
                Felt::new(BigInteger256([
                    0x8c7dbfae9df80d7d,
                    0xac6da4003b1f50a1,
                    0x849a8027eb124490,
                    0x61b22c4790f4c974,
                ])),
                Felt::new(BigInteger256([
                    0xb23bee899fef65a1,
                    0x0edb16e2f913e1d6,
                    0x195848831ab549d9,
                    0x37840337e2698868,
                ])),
                Felt::new(BigInteger256([
                    0x9390eb6c4486974a,
                    0x379b9771576f5061,
                    0x3fa6fe16dd80c41c,
                    0x1b149077ad14a121,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xae6badcc3ada1572,
                    0x21e98d3b9dfe6d96,
                    0xe6c384826cb60623,
                    0x3f81b48946b19349,
                ])),
                Felt::new(BigInteger256([
                    0x4e9a0a258e5c0e3d,
                    0x2ec4b1581be0c27e,
                    0xd424f61c841ce823,
                    0x16a444dd7d99034e,
                ])),
                Felt::new(BigInteger256([
                    0x258cd140a5e8dad4,
                    0x47c65f3a0865deee,
                    0x90a501601002913b,
                    0x4963d39bf5a6ff39,
                ])),
                Felt::new(BigInteger256([
                    0xf1f63067b95e5c0d,
                    0xf4866f8be6dee7e5,
                    0x1609234220d4c22c,
                    0x4c3927e7580e306a,
                ])),
                Felt::new(BigInteger256([
                    0x2eced3aaf298f493,
                    0x1fe617eb7123541c,
                    0x1ce65e7981eba746,
                    0x0d0c867401ddaec8,
                ])),
                Felt::new(BigInteger256([
                    0x927303152fc6b46d,
                    0x93ac39e8417dcc01,
                    0x96ffff85f4070de8,
                    0x019e6e2e0ee30abf,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf5dee7fbd28966c6,
                    0x8ff94500246c49b6,
                    0x7041feb895cf61da,
                    0x1f6693f7283f7b45,
                ])),
                Felt::new(BigInteger256([
                    0x901a55ac8d9227a2,
                    0xe960bd9e78fa0e7c,
                    0x937c3c38de45ac80,
                    0x0ebc4b2c3754808e,
                ])),
                Felt::new(BigInteger256([
                    0x759483c10ec693d2,
                    0xdd66ceddc36e813e,
                    0x599bc92591ea7212,
                    0x5db4a379f0c55565,
                ])),
                Felt::new(BigInteger256([
                    0x73ab97d055cda522,
                    0xdab4ef3ccaf78eab,
                    0x839d46e00ec3192e,
                    0x6de8cc51d858341b,
                ])),
                Felt::new(BigInteger256([
                    0x34b4b9956ab72ded,
                    0xda60f07cdac46034,
                    0x4407184cace23809,
                    0x5c03e4c2f2a6f1ff,
                ])),
                Felt::new(BigInteger256([
                    0xb28b4abff176298d,
                    0xf0cfb56319f90416,
                    0xdc8bdc8cb523e51b,
                    0x6af55b10610b1d40,
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
                    0x15afbb12b73d98e1,
                    0xdc3976de9fa9030e,
                    0xc9455d51c5b825c6,
                    0x3a14009957c1ed09,
                ])),
                Felt::new(BigInteger256([
                    0xcbac2ad65630549a,
                    0x8436bc0871496521,
                    0xcb52096792edeaf6,
                    0x670ef5e0f1263bac,
                ])),
                Felt::new(BigInteger256([
                    0x13fd06a4f172b8e1,
                    0x68fc5bb6c8d2f799,
                    0x92e492592a860532,
                    0x026f8d32a3fdd433,
                ])),
                Felt::new(BigInteger256([
                    0xc695d6a70cd2ef44,
                    0xa426e73f56222642,
                    0x74452d872a97aa9f,
                    0x075d570b41994fae,
                ])),
                Felt::new(BigInteger256([
                    0x25caa6f4bc565aa1,
                    0xd69e67660a6a7e38,
                    0x046006676b2dbd9b,
                    0x6a0e73da4a04c120,
                ])),
                Felt::new(BigInteger256([
                    0x89a06825d00f769e,
                    0xe0ce397910a19b4f,
                    0xfb3e61df8598cfa7,
                    0x22b8357c1fb88479,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x22f35ff5b7b8d2cc,
                    0x7697180a5ecd8851,
                    0x7974b4eebd33d876,
                    0x5bbc052e2dee41a6,
                ])),
                Felt::new(BigInteger256([
                    0xa082193c52ac1873,
                    0xb3d24313708c5e34,
                    0x2a19e7196181d927,
                    0x0000ef8732607d06,
                ])),
                Felt::new(BigInteger256([
                    0x045dd748e456241b,
                    0x48e9e8715ee4cc64,
                    0xbc90ad429b1e9cac,
                    0x0ecd13eaed166e2e,
                ])),
                Felt::new(BigInteger256([
                    0x8f8188ba307d3255,
                    0x5a01c85334ec1951,
                    0x1bd2be6e19b94fb7,
                    0x000bbd7b59cfb502,
                ])),
                Felt::new(BigInteger256([
                    0xd0dca4bc4a2ef83d,
                    0xfc887d7790ff0ac3,
                    0x87bdda08f4d9e6cb,
                    0x0ba0d0250cada18d,
                ])),
                Felt::new(BigInteger256([
                    0xb0789ff8552c6bde,
                    0x70332ec570dcec99,
                    0x38b242ad2a25877d,
                    0x221a27d79200679a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9c71e7f286079d07,
                    0x261c69257e486d1b,
                    0x998760a9f4509f9a,
                    0x069b01d05840b537,
                ])),
                Felt::new(BigInteger256([
                    0xe0d09ffcbf519a2e,
                    0x2cb8cea96e27222d,
                    0x057f354fa65621c8,
                    0x4fbe00a6279b801a,
                ])),
                Felt::new(BigInteger256([
                    0xffe746138aef7357,
                    0x1d7a339e0d8853b2,
                    0x04352f9c8c742934,
                    0x6066904fab3bcceb,
                ])),
                Felt::new(BigInteger256([
                    0xaa3e93b93b2d497f,
                    0x7bad9cc5f67c1358,
                    0xb9d72f6d6c46c8a0,
                    0x5d72f4397228b7be,
                ])),
                Felt::new(BigInteger256([
                    0x059ba7ac3abddef6,
                    0x57d5c4641e599851,
                    0xe446b4b49e761048,
                    0x57a2bf9b1e15827f,
                ])),
                Felt::new(BigInteger256([
                    0xa04d33acd9d8cb29,
                    0x04b9f34e087a6b95,
                    0xbaefd4d23857fec6,
                    0x4f3120ba879889fc,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7f05630698beab25,
                    0x20524acb69cc9c9a,
                    0x3bff1d1889f1aa06,
                    0x48a0182593e1ea67,
                ])),
                Felt::new(BigInteger256([
                    0xbc689e608d421021,
                    0x504d574da9fe35fb,
                    0x4d2961b5e247bdba,
                    0x2ef7af2f53b59d8e,
                ])),
                Felt::new(BigInteger256([
                    0xaacd1e82423b6a43,
                    0xce4c4625bf24e18a,
                    0x419102e2f8dba6c5,
                    0x6c0eaf45db3557ce,
                ])),
                Felt::new(BigInteger256([
                    0x1df2cadfd5e448ae,
                    0x143169a20d1b3194,
                    0xb62551bd6929be0b,
                    0x56250b8fb98a997d,
                ])),
                Felt::new(BigInteger256([
                    0xc0555bceb37e54ca,
                    0x93eb0d8626b02560,
                    0x30dfa8193353ac13,
                    0x2505e7bea786decb,
                ])),
                Felt::new(BigInteger256([
                    0x69f73f5910d77f79,
                    0x221d012795eb444b,
                    0x19670fe08852f042,
                    0x6a0242f0282d9bae,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb9bb72e4fc1b718e,
                    0x5ba3e1fe6a95837d,
                    0x6b60ee79db5bc6c4,
                    0x417413da438ac168,
                ])),
                Felt::new(BigInteger256([
                    0x540e4e98a1e9c9f9,
                    0x64a343d45e90b273,
                    0xe21fbfc19d98de72,
                    0x33d1bf9b6b947d6b,
                ])),
                Felt::new(BigInteger256([
                    0xb8b35e87e2758ba5,
                    0x993f726999115dba,
                    0xde0ff827633da258,
                    0x00166f04dab32a2c,
                ])),
                Felt::new(BigInteger256([
                    0x895e81da5662da1a,
                    0x7a9540b5be24c750,
                    0x39fd3cd6b93ef5a6,
                    0x0ec53aad01ab3c87,
                ])),
                Felt::new(BigInteger256([
                    0xa1ad5a9556ed88be,
                    0x18cf6caebc9c731e,
                    0x5924b7b3d651a643,
                    0x4bd429b0adb04b1b,
                ])),
                Felt::new(BigInteger256([
                    0xb859cfa3f560836d,
                    0xb94e50315f933053,
                    0xe6fe7e833e00b0e7,
                    0x73ddf74e4d4d0386,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf647d496501a5a70,
                    0x5342cccc27c39ba2,
                    0x8b7e2959e25cabdf,
                    0x5bea4a55d846be8f,
                ])),
                Felt::new(BigInteger256([
                    0x550efa8a523e2421,
                    0xc5f57266eded53d7,
                    0xcde6b65d9a6adadc,
                    0x3b32db878f2fe2ac,
                ])),
                Felt::new(BigInteger256([
                    0x72305e79200161f0,
                    0xab7383078a43e85f,
                    0x3354344f5cbcfb18,
                    0x45a639611ac594ba,
                ])),
                Felt::new(BigInteger256([
                    0x341c911d048e0387,
                    0x4c276cef1e1f4d38,
                    0x0911f62951a5d0bc,
                    0x5861240cbd6cea35,
                ])),
                Felt::new(BigInteger256([
                    0xfa6e373c277bb9c2,
                    0xaabce789f079d5dd,
                    0x24237c40e54361de,
                    0x244ec82a2fd7b125,
                ])),
                Felt::new(BigInteger256([
                    0xc7f47145daba1e0b,
                    0xad8081bbb7750970,
                    0x392468f575aa6a38,
                    0x12d8f758e90e0301,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(); 6],
            [
                Felt::new(BigInteger256([
                    0x00000024ffffffdb,
                    0xe5974b91003cb425,
                    0x98a3c6d69b9bc73a,
                    0x3ea6d0fafc3ce490,
                ])),
                Felt::new(BigInteger256([
                    0x00000012ffffffed,
                    0xc8ecd3c7001f2c13,
                    0x32b4f76748fcf79a,
                    0x655c94d3e94fb3a4,
                ])),
                Felt::new(BigInteger256([
                    0x00000012ffffffed,
                    0xc8ecd3c7001f2c13,
                    0x32b4f76748fcf79a,
                    0x655c94d3e94fb3a4,
                ])),
                Felt::new(BigInteger256([
                    0x00000024ffffffdb,
                    0xe5974b91003cb425,
                    0x98a3c6d69b9bc73a,
                    0x3ea6d0fafc3ce490,
                ])),
                Felt::new(BigInteger256([
                    0x00000012ffffffed,
                    0xc8ecd3c7001f2c13,
                    0x32b4f76748fcf79a,
                    0x655c94d3e94fb3a4,
                ])),
                Felt::new(BigInteger256([
                    0x00000012ffffffed,
                    0xc8ecd3c7001f2c13,
                    0x32b4f76748fcf79a,
                    0x655c94d3e94fb3a4,
                ])),
            ],
            [
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::new(BigInteger256([
                    0x00000024ffffffdb,
                    0xe5974b91003cb425,
                    0x98a3c6d69b9bc73a,
                    0x3ea6d0fafc3ce490,
                ])),
                Felt::new(BigInteger256([
                    0x00000012ffffffed,
                    0xc8ecd3c7001f2c13,
                    0x32b4f76748fcf79a,
                    0x655c94d3e94fb3a4,
                ])),
                Felt::new(BigInteger256([
                    0x00000012ffffffed,
                    0xc8ecd3c7001f2c13,
                    0x32b4f76748fcf79a,
                    0x655c94d3e94fb3a4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x00000024ffffffdb,
                    0xe5974b91003cb425,
                    0x98a3c6d69b9bc73a,
                    0x3ea6d0fafc3ce490,
                ])),
                Felt::new(BigInteger256([
                    0x00000012ffffffed,
                    0xc8ecd3c7001f2c13,
                    0x32b4f76748fcf79a,
                    0x655c94d3e94fb3a4,
                ])),
                Felt::new(BigInteger256([
                    0x00000012ffffffed,
                    0xc8ecd3c7001f2c13,
                    0x32b4f76748fcf79a,
                    0x655c94d3e94fb3a4,
                ])),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0x191238989bb2e2a5,
                    0x0b311ca4b5316e60,
                    0xac804e96e4b60aa6,
                    0x07871fa0ff10d32e,
                ])),
                Felt::new(BigInteger256([
                    0x6d47146ca790fba1,
                    0xeb9910e38eb8d160,
                    0x659d8f2178ae5d18,
                    0x3e422b899b3b78d7,
                ])),
                Felt::new(BigInteger256([
                    0x77774f024a523b9e,
                    0xa3cec7c997c20221,
                    0x1234c8dcfef59883,
                    0x3053e9f854fb9604,
                ])),
                Felt::new(BigInteger256([
                    0xeca4550c1959c5be,
                    0x190b8e8c1513df2e,
                    0x26994113e6b55159,
                    0x0ea9520f2fbd20ac,
                ])),
                Felt::new(BigInteger256([
                    0x1d83edace62a5c1a,
                    0xe0bf4f9775fcc95c,
                    0xfa48cef1114a5f9b,
                    0x4c6663520b50f616,
                ])),
                Felt::new(BigInteger256([
                    0x58c0cf84033ee043,
                    0x6cd81c70afb70d21,
                    0x5ac8ac0a64a5996a,
                    0x58e8cb8c6dc22c57,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdb0bd3383323cfa4,
                    0x65aacadc5e2a7fe4,
                    0x73b0106be0a89a18,
                    0x28ca2509e7378eb6,
                ])),
                Felt::new(BigInteger256([
                    0xe2065c3148bfe7fb,
                    0xd5111234679d2142,
                    0x984980d252ea224e,
                    0x4f6ad8ceb24e44ab,
                ])),
                Felt::new(BigInteger256([
                    0x998790423d10001d,
                    0x98299fbe671818d5,
                    0x38ba4ebaf8e228f6,
                    0x4d4de31590e74459,
                ])),
                Felt::new(BigInteger256([
                    0xb36a0bad2a8dc06d,
                    0xd0c7b918a037b145,
                    0x21fd2e5d951d6392,
                    0x0b90ed879c4d9ecf,
                ])),
                Felt::new(BigInteger256([
                    0x6de001cbf2c7c46e,
                    0xe2c826837450a898,
                    0x833351b8d3109c4c,
                    0x2e0d265c135bfc36,
                ])),
                Felt::new(BigInteger256([
                    0xf602a9d88cf267de,
                    0xb1f0655d9cc25145,
                    0x0ab5f759f5d426c8,
                    0x738b9656448f877b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc39a1034470a1d17,
                    0xff3e38b0ccb8a4a9,
                    0x8ccecf4b690f8014,
                    0x5c4afe6020312435,
                ])),
                Felt::new(BigInteger256([
                    0x7c95727e11e55e90,
                    0x2abac90f4b33b134,
                    0x891fd3113808d1af,
                    0x419508b134cde00d,
                ])),
                Felt::new(BigInteger256([
                    0x27d53db2f47658b5,
                    0x013c3e4aefac15a3,
                    0x092e3189d75cd02e,
                    0x6a73f65512fec442,
                ])),
                Felt::new(BigInteger256([
                    0x1f1f0ee589320ec4,
                    0x6033a877af3d1cf0,
                    0x11eefd7a1106cdc7,
                    0x14cd4025ec8aff47,
                ])),
                Felt::new(BigInteger256([
                    0x4d9ee570b2d3ac91,
                    0x741f8506e444075a,
                    0x4d83ed4c8950a34c,
                    0x0978faa1a17ca5ba,
                ])),
                Felt::new(BigInteger256([
                    0x71cd5d21b0372d5c,
                    0xf0ce16bed373812b,
                    0x7f5ae4ffc00e2940,
                    0x5a85663ed2a6e389,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0afcaab36512bb55,
                    0xd85e2eb2f19dd731,
                    0xd2f44131851c23d9,
                    0x654a12a4d90dcfd7,
                ])),
                Felt::new(BigInteger256([
                    0xe709d6fdf5a0a314,
                    0xca86110c4dd87967,
                    0xed8aaacaf6ce0f03,
                    0x407efef7c2be819f,
                ])),
                Felt::new(BigInteger256([
                    0xe05b7215fcb42862,
                    0x5e2574f44dc393c4,
                    0x3292f81c7095d290,
                    0x53c2c2dc6a04ebc6,
                ])),
                Felt::new(BigInteger256([
                    0x5c3874d65bec6b30,
                    0x7155f64e3454bffb,
                    0x8486464d158da907,
                    0x2ffefb76ba2ff66b,
                ])),
                Felt::new(BigInteger256([
                    0xfbf0274c9d93d0ff,
                    0x4cf01a0a18639cbe,
                    0x11f0e3f761ffbe83,
                    0x32798fa9e8cdbd38,
                ])),
                Felt::new(BigInteger256([
                    0xca3f8ce2cf3019aa,
                    0x4682dd6bb1dc6787,
                    0x8bfc72363be0dac6,
                    0x6787aebcf98fd4e1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe784da039671b38d,
                    0xbcc757087bce4c3a,
                    0x60bf94ab6bdcc744,
                    0x706f3947b70de4f5,
                ])),
                Felt::new(BigInteger256([
                    0x1ab15735cf3c0d09,
                    0x9d45a2b2f8a16a0d,
                    0x2cb69f4726023d9d,
                    0x01f535448067e8c6,
                ])),
                Felt::new(BigInteger256([
                    0x20e1d167691f707c,
                    0x30675426e1bf38a2,
                    0xe2eedd1dd9d19014,
                    0x2e5e1c4b789cfc52,
                ])),
                Felt::new(BigInteger256([
                    0x10b8b3285fe39a26,
                    0x0a443f4035a87fd4,
                    0x19793a9f7718a80f,
                    0x1d2986f5a4dad3a2,
                ])),
                Felt::new(BigInteger256([
                    0x1b9cb733a90202df,
                    0x84b739d24f345ea8,
                    0x6f9c300611c75ead,
                    0x3f3b6d13b370fbc4,
                ])),
                Felt::new(BigInteger256([
                    0xad75cb98ac421ab4,
                    0x3edb559f4607517f,
                    0x90db891ca351fa49,
                    0x51cc11303eb6db64,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x98d09310d31c0716,
                    0x2281e4e37e3b7ff5,
                    0x90fd5b4f2943ca62,
                    0x4c80c8ac5dce1ae0,
                ])),
                Felt::new(BigInteger256([
                    0x6aa9647582622c1c,
                    0x26aca058dd947e1b,
                    0xc09115bad5c92c4e,
                    0x3b0472e552ca3fea,
                ])),
                Felt::new(BigInteger256([
                    0x83362925a2f7ff1b,
                    0xc1cab6f18e9455b1,
                    0x9e52fbf1ede578f0,
                    0x4caf314e9a33bda2,
                ])),
                Felt::new(BigInteger256([
                    0x3c4ab4173b08084b,
                    0xc6b6056c2c4b422a,
                    0x09013c06dfe53ee5,
                    0x58eb1e770646f391,
                ])),
                Felt::new(BigInteger256([
                    0x2f2aa0522217f079,
                    0xca9d2fc07ad22fdd,
                    0x9ca4682f664d4922,
                    0x5e2b773c76cca930,
                ])),
                Felt::new(BigInteger256([
                    0xd514850af3aa35dd,
                    0xa999576768ee1bbd,
                    0xa5813cccef425ffb,
                    0x0d989e9378884d61,
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
