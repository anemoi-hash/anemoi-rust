//! Implementation of the Anemoi permutation

use super::{mul_by_generator, sbox, BigInteger256, Felt};
use crate::{Jive, Sponge};
use ark_ff::{Field, One, Zero};
use unroll::unroll_for_loops;

/// Digest for Anemoi
mod digest;
/// Sponge for Anemoi
mod hasher;

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

/// The number of rounds is set to 13 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 13;

// HELPER FUNCTIONS
// ================================================================================================

/// Applies the Anemoi S-Box on the current
/// hash state elements.
#[inline(always)]
pub(crate) fn apply_sbox_layer(state: &mut [Felt; STATE_WIDTH]) {
    let mut x: [Felt; NUM_COLUMNS] = state[..NUM_COLUMNS].try_into().unwrap();
    let mut y: [Felt; NUM_COLUMNS] = state[NUM_COLUMNS..].try_into().unwrap();

    x.iter_mut().enumerate().for_each(|(i, t)| {
        let y2 = y[i].square();
        *t -= mul_by_generator(&y2);
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
        *t += mul_by_generator(&y2) + sbox::DELTA;
    });

    state[..NUM_COLUMNS].copy_from_slice(&x);
    state[NUM_COLUMNS..].copy_from_slice(&y);
}

/// Applies matrix-vector multiplication of the current
/// hash state with the Anemoi MDS matrix.
#[inline(always)]
pub(crate) fn apply_linear_layer(state: &mut [Felt; STATE_WIDTH]) {
    state[0] += mul_by_generator(&state[1]);
    state[1] += mul_by_generator(&state[0]);

    state[3] += mul_by_generator(&state[2]);
    state[2] += mul_by_generator(&state[3]);
    state.swap(2, 3);

    // PHT layer
    state[2] += state[0];
    state[3] += state[1];

    state[0] += state[2];
    state[1] += state[3];
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

    apply_linear_layer(state)
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

    apply_linear_layer(state);
    apply_sbox_layer(state);
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
                    0x11113b40b05a7e48,
                    0x9fb3dabf869e22cf,
                    0xc8e111328e434c88,
                    0x034528e412510179,
                ])),
                Felt::new(BigInteger256([
                    0xa89eac61d15166b6,
                    0x3b3efe3c11149c03,
                    0x3f2dd0588818092c,
                    0x0121a1fcb1d5d51b,
                ])),
                Felt::new(BigInteger256([
                    0x4be8fc77c3aecedd,
                    0x34c9f56d91e1c548,
                    0x3c37bf0bcfc92347,
                    0x10a38a8834a9e581,
                ])),
                Felt::new(BigInteger256([
                    0x3ea7419d26a1ef2e,
                    0x8c2390e67cb6f1fa,
                    0x8c936398b923bd64,
                    0x004b45b01192528e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf8dcbb109af72c46,
                    0x3482b0b9a1e0931b,
                    0x1480ba98aed7dbe3,
                    0x08777eec776232df,
                ])),
                Felt::new(BigInteger256([
                    0x437fc84929d49464,
                    0xad6bafed2c9f55da,
                    0x5169e3c78d268c9d,
                    0x103e8e58770a1dd8,
                ])),
                Felt::new(BigInteger256([
                    0x209fb5afe6fb56de,
                    0x021ef0cdb389c867,
                    0x9edee0de62142039,
                    0x0e55986619a26417,
                ])),
                Felt::new(BigInteger256([
                    0x5554dc96fc66c45d,
                    0x14dc675e2ae4c45b,
                    0x7ddfbcac5e9794ca,
                    0x01823ae3a3fd0e78,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xeaf1522f0c0904dd,
                    0xec317034516fb055,
                    0x7188e97a2aff041d,
                    0x0587e52a58eacbcc,
                ])),
                Felt::new(BigInteger256([
                    0x924a21dfc1b66e56,
                    0xce09d068b6f5fbe4,
                    0x897acfce39d9fbd1,
                    0x0ee87a8b11348d2b,
                ])),
                Felt::new(BigInteger256([
                    0x55ca5f136aa16e8f,
                    0xe6ae1840c0c67699,
                    0x5663533c814c0fe0,
                    0x0952873a201f6219,
                ])),
                Felt::new(BigInteger256([
                    0x491579b572db8f45,
                    0xea5ec6235bc6a29d,
                    0xc54de216f3c08fc5,
                    0x0ff982c56ce28688,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x137ffdf9cfc15d2e,
                    0xe198845fc163c28e,
                    0x7bf6621f23836c51,
                    0x01828741375b9eb3,
                ])),
                Felt::new(BigInteger256([
                    0xda1366cd8352c32e,
                    0xe8f0d7caa5ec6b68,
                    0x3b97e014347fca9e,
                    0x085c78b4adfef891,
                ])),
                Felt::new(BigInteger256([
                    0xff41b834db987da7,
                    0x4b0d218b35e9cc41,
                    0xfd9e58ec62b612e6,
                    0x0549ee22cc825537,
                ])),
                Felt::new(BigInteger256([
                    0x93fb19e5b507bae9,
                    0x8fd4a4c7378dfa76,
                    0x5cf2e7ac69b7ba4f,
                    0x10b7e5731a6d6300,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x82fad2c055085932,
                    0xd233bf2b054c4935,
                    0x48c49d65a600bd44,
                    0x0b9cf01d9b7b470d,
                ])),
                Felt::new(BigInteger256([
                    0x7320436e8230855e,
                    0x1dc9b22a57e4ed14,
                    0x6a1cb49c365bb690,
                    0x0267969c4692c0f4,
                ])),
                Felt::new(BigInteger256([
                    0x5cd3697adef8fd78,
                    0xb467ae300db7091d,
                    0x421e7f6982fadc33,
                    0x091d513ad0822bbe,
                ])),
                Felt::new(BigInteger256([
                    0x1d594ebb579d95b3,
                    0x2ef30459e50dd0cb,
                    0xf8b7ab26d85af6e3,
                    0x1054fd2789613a36,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe4c41ed3fc54efeb,
                    0x3ced652899fa6d34,
                    0x7ca55f105b2ebb48,
                    0x0babcc2f75073efb,
                ])),
                Felt::new(BigInteger256([
                    0xb9a4501cc39f6020,
                    0xcc3b6797a3b84d59,
                    0x3ec0910471847637,
                    0x09c620b6a78d6b78,
                ])),
                Felt::new(BigInteger256([
                    0x2bcd7f4e2a1a8192,
                    0xd3ec15945342093e,
                    0x9c0be9f2e40dad9b,
                    0x01c96c6e54612a45,
                ])),
                Felt::new(BigInteger256([
                    0x630f4b3f1ee8211b,
                    0x9046e062204d4eb7,
                    0x83b30944251ad91e,
                    0x09c77452c13bcaa2,
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
                Felt::new(BigInteger256([
                    0xb76f9745d1745d17,
                    0xfed18274afffffff,
                    0xfce619835b36a173,
                    0x068b6ffd78dc8d16,
                ])),
                Felt::zero(),
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
                Felt::new(BigInteger256([
                    0x8cf500000000000e,
                    0xe75281ef6000000e,
                    0x49dc37a90b0ba012,
                    0x055f8b2c6e710ab9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb00ff5a6ecffa074,
                    0x3e4e99ad7d45eefa,
                    0x0e3220cc3f90b83b,
                    0x014d00b9ee63a8d1,
                ])),
                Felt::new(BigInteger256([
                    0xd8b7eeb059995c47,
                    0xe3cc7bbfaa035ca8,
                    0x1eb284881c0bd6f9,
                    0x0198c6b5b09c000f,
                ])),
                Felt::new(BigInteger256([
                    0x1b1bce39bff73e23,
                    0x4c23457497815fcf,
                    0x23bc3f04759e9029,
                    0x0b32f10e5c980987,
                ])),
                Felt::new(BigInteger256([
                    0x5de4ec02a1d58418,
                    0x83f0cf4e33273866,
                    0x8da1fbeb4e0e208e,
                    0x043079d338844f75,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x88d9ef1fdc044d8f,
                    0x4e94166ef875b883,
                    0xbaf101097d697a23,
                    0x0621639a8d73c4fb,
                ])),
                Felt::new(BigInteger256([
                    0x6b1b0fd1682ffd5a,
                    0x2da191a3f0533235,
                    0xfb7e0b02726712a3,
                    0x08332d095fe5fcb1,
                ])),
                Felt::new(BigInteger256([
                    0x0756b7daab757774,
                    0xceb6bc9a846f274f,
                    0xc1c62f432849f803,
                    0x026ebf51d74c1ac7,
                ])),
                Felt::new(BigInteger256([
                    0xb729f9f723e654e0,
                    0xcc8a8cae44cebf51,
                    0x89596225b9821e84,
                    0x03ad0aaa75195795,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1c419bd9cc8cf329,
                    0xfaf478e24c9aa387,
                    0x458cfc6a4e517d28,
                    0x0f5bd34daf33c79c,
                ])),
                Felt::new(BigInteger256([
                    0x73b78a1b3baabc6c,
                    0x0d36e01531427e02,
                    0x74d29707ede9d007,
                    0x0166c35ce690e95b,
                ])),
                Felt::new(BigInteger256([
                    0x69db80c52031903a,
                    0xea7ea79456497883,
                    0xea1db66acaa1e33d,
                    0x0b20f6f3597b9575,
                ])),
                Felt::new(BigInteger256([
                    0x7c49997312c29676,
                    0x96b598b7cb72d1a0,
                    0x6dff4453edf81b57,
                    0x0e7bc746f45ca4e2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x45fb87e6f67a47e0,
                    0xf060a776d8810002,
                    0xdd78ab7c9a00d68e,
                    0x01e75cb636104f49,
                ])),
                Felt::new(BigInteger256([
                    0x8d550feb3f692770,
                    0x2cd82e278fcb052b,
                    0x0b015451c45604b7,
                    0x0a53073980555328,
                ])),
                Felt::new(BigInteger256([
                    0x6434aaed9339701b,
                    0x3da5d84f35bceacb,
                    0xe3406949a723a1ad,
                    0x0ab91530f87bf858,
                ])),
                Felt::new(BigInteger256([
                    0x7472179591e83a42,
                    0xfd528537935703b0,
                    0xd8f322f9814c9781,
                    0x03f56e5bce44d92e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6fc7e8519b5da1fc,
                    0xea4e5f74a1567715,
                    0x30b5431b1048fe6b,
                    0x0cb55bed98922d49,
                ])),
                Felt::new(BigInteger256([
                    0x2761dea00cf783f6,
                    0xad730b46b78858ec,
                    0x8d84b9718e54ff5d,
                    0x114af1c449954dc5,
                ])),
                Felt::new(BigInteger256([
                    0xace307d174041766,
                    0x4d683fc34cc16ae7,
                    0x307f3857e9e484d4,
                    0x034b37329d91dba7,
                ])),
                Felt::new(BigInteger256([
                    0x9708e2f1b61fd454,
                    0x5a44f7b6485aa9a0,
                    0x14f50f449a9e6a24,
                    0x096acae0f45bbd78,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x59e0831bd4c424af,
                    0x79c2292d4eee8fb1,
                    0xc1f67dd659c8f47e,
                    0x0ab13e0cf3834a68,
                ])),
                Felt::new(BigInteger256([
                    0x5e53992fa4659167,
                    0x4d563f7d9126fb1a,
                    0xe4e42bc83ac74a56,
                    0x02749b1808ae7168,
                ])),
                Felt::new(BigInteger256([
                    0x8c2e91f60e0d92a4,
                    0x7da89252b743eeb0,
                    0x74a5a387c7e3182f,
                    0x117d697dc9f8aa4a,
                ])),
                Felt::new(BigInteger256([
                    0xa95e4e59b1f361c4,
                    0x7c1a48e10a48d99e,
                    0x5e2ac4c26886427d,
                    0x0fee289d3ccf88a0,
                ])),
            ],
        ];

        for i in input.iter_mut() {
            apply_sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
