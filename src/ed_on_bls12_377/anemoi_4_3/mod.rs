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

/// The number of rounds is set to 11 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 11;

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
                    0x9eea8b76810542eb,
                    0x72b4aa416595316c,
                    0xa6aa981b18f309dc,
                    0x0cd2902d03bc35d8,
                ])),
                Felt::new(BigInteger256([
                    0x31ce84ae52e457b6,
                    0x3708a47544d0865c,
                    0xa92152ec814cdb77,
                    0x0dab47756959ed92,
                ])),
                Felt::new(BigInteger256([
                    0xeda17a3a331b8950,
                    0xdd62a33eab954fb4,
                    0xd8b4c39c8cdbbddf,
                    0x11db5a7b89a527ec,
                ])),
                Felt::new(BigInteger256([
                    0x12ec801b8013d936,
                    0xf30915b03c85d25e,
                    0x404ea13815d84098,
                    0x1292d4d706bdef29,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb505cac7e88c477e,
                    0xef252c76155e6e1a,
                    0x6238ec899b9a6d7d,
                    0x0d09de2389eddf5a,
                ])),
                Felt::new(BigInteger256([
                    0x087cc42d7b60fe6e,
                    0x970a3ee9fcecc968,
                    0x190f6f0bc65d2dcd,
                    0x052fdf069ca14b04,
                ])),
                Felt::new(BigInteger256([
                    0x74a7e4de218ce201,
                    0x1bb934c9f0f00807,
                    0x77294d68d5b00eed,
                    0x0951bc131f5c0415,
                ])),
                Felt::new(BigInteger256([
                    0x4ad6041927d63c1c,
                    0x7b59db95a1cffcab,
                    0x7056447f8b65f76b,
                    0x0bc0b627828378fc,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9bd7c12405dff86c,
                    0xc1d06ede876d6b5a,
                    0x859e5970e52746ac,
                    0x0cd2db0be940896e,
                ])),
                Felt::new(BigInteger256([
                    0x56ef1b46698adeca,
                    0x6b3a71b5e7d8dece,
                    0xc432eba635cc51f3,
                    0x0297d233dc2c876d,
                ])),
                Felt::new(BigInteger256([
                    0x73c7e016fa64ccaf,
                    0xde29c73ebda48ad8,
                    0xf68a847741e6d99e,
                    0x086db5b41e35b6ba,
                ])),
                Felt::new(BigInteger256([
                    0x9c504d649ca5fe7b,
                    0xbbc62eaa9b58c4f6,
                    0x5f4490a1f2f8f27a,
                    0x0821a3fa8f1f8f91,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1f4400849a31ac9b,
                    0x6ca3f62b3d11063f,
                    0x3d28f75866f8d16b,
                    0x0ad629ce60257a24,
                ])),
                Felt::new(BigInteger256([
                    0xbb7868fb7eb69da5,
                    0x3655688bcecdbe7c,
                    0xa201d9f4bbf7e12b,
                    0x0bc2985d34086306,
                ])),
                Felt::new(BigInteger256([
                    0x555a08513b8db314,
                    0xecfbe4c1951ecd2a,
                    0x4ae2ac1e3022cceb,
                    0x10ea43defde10f93,
                ])),
                Felt::new(BigInteger256([
                    0xeb3b71f931685fc8,
                    0x24c4aa25cfba3236,
                    0x0010cce0f639a89f,
                    0x0abae805b083de10,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x88b6168ec75cfd88,
                    0x47395d0d69dfece3,
                    0x9efec02b345f4634,
                    0x0f2713faccf13ff3,
                ])),
                Felt::new(BigInteger256([
                    0xd55021a6a9dd3c4d,
                    0x11ebaea02da86041,
                    0x4dcaca3baa2ae284,
                    0x0647670ff604faae,
                ])),
                Felt::new(BigInteger256([
                    0x6520318c20369958,
                    0x3d44dad06f50576f,
                    0x8f4cbeafade8f795,
                    0x11d746926566dee0,
                ])),
                Felt::new(BigInteger256([
                    0x766cf64b7784be65,
                    0x44b133d68d62d30c,
                    0xa1f1c78d66e364c8,
                    0x056dd9232be6d182,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x954b85c181a65a61,
                    0x0fd29214bf3f5dd9,
                    0x8c3b2be9128ad828,
                    0x014d25a3abf44a6c,
                ])),
                Felt::new(BigInteger256([
                    0x96eba51de9790a48,
                    0x0216be36dd06a2e6,
                    0x30e1d7a24626690d,
                    0x06ecc50a39d07af2,
                ])),
                Felt::new(BigInteger256([
                    0x955ce504b9a8032a,
                    0xb6008b5c0bfe0b60,
                    0x26a450e819ba93f6,
                    0x11a4ddf29ddccf44,
                ])),
                Felt::new(BigInteger256([
                    0x0822733da557dd13,
                    0x348914bd9d8feeff,
                    0x3ab0e982b79aba15,
                    0x0135191eb1694307,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(); 4],
            [
                Felt::new(BigInteger256([
                    0x9c777ffffffffec5,
                    0xab3f94760ffffeb8,
                    0x02251ba4877a6e56,
                    0x071a44984b108eb7,
                ])),
                Felt::new(BigInteger256([
                    0x94c3ffffffffe4d8,
                    0x02d0883f7fffe3c6,
                    0xdfb1bf87b7bc5b55,
                    0x01872ef533960e4d,
                ])),
                Felt::new(BigInteger256([
                    0x9c777ffffffffec5,
                    0xab3f94760ffffeb8,
                    0x02251ba4877a6e56,
                    0x071a44984b108eb7,
                ])),
                Felt::new(BigInteger256([
                    0x94c3ffffffffe4d8,
                    0x02d0883f7fffe3c6,
                    0xdfb1bf87b7bc5b55,
                    0x01872ef533960e4d,
                ])),
            ],
            [
                Felt::zero(),
                Felt::zero(),
                Felt::new(BigInteger256([
                    0x9c777ffffffffec5,
                    0xab3f94760ffffeb8,
                    0x02251ba4877a6e56,
                    0x071a44984b108eb7,
                ])),
                Felt::new(BigInteger256([
                    0x94c3ffffffffe4d8,
                    0x02d0883f7fffe3c6,
                    0xdfb1bf87b7bc5b55,
                    0x01872ef533960e4d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9c777ffffffffec5,
                    0xab3f94760ffffeb8,
                    0x02251ba4877a6e56,
                    0x071a44984b108eb7,
                ])),
                Felt::new(BigInteger256([
                    0x94c3ffffffffe4d8,
                    0x02d0883f7fffe3c6,
                    0xdfb1bf87b7bc5b55,
                    0x01872ef533960e4d,
                ])),
                Felt::zero(),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0x4591f271a0a4cc7f,
                    0x92cb5c684f80bd48,
                    0x2442e6887212e605,
                    0x0ed45e5a6eac4b0d,
                ])),
                Felt::new(BigInteger256([
                    0x771e5a72210dea8e,
                    0x8686378179e0ca7f,
                    0xfa33b683d3023fd9,
                    0x03de42941504c0a0,
                ])),
                Felt::new(BigInteger256([
                    0xa14c011be471a600,
                    0x44dce32d1b5aabd3,
                    0x9058d01043f171b8,
                    0x00b1e5539b192817,
                ])),
                Felt::new(BigInteger256([
                    0xc018129fd4dfcd4f,
                    0x6eb3b220356013e3,
                    0xdfa257e40763d3b4,
                    0x0e79aa4c43a1f49c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3355a6b082e224ec,
                    0xd007ca98f1b7bd05,
                    0x4552a8d67e4e3d20,
                    0x0f22a87d62be71ae,
                ])),
                Felt::new(BigInteger256([
                    0xbc9e1758bad02aa4,
                    0x29b94a2422b707c7,
                    0x417c8558252c0e87,
                    0x021d392641db6feb,
                ])),
                Felt::new(BigInteger256([
                    0xe2832f3009f1a827,
                    0x02f147fd6670ad43,
                    0x86239b33f222afbd,
                    0x0b6984bb9480b81e,
                ])),
                Felt::new(BigInteger256([
                    0x690a72fefc51554e,
                    0xcecd59a02e9eebcf,
                    0x1510b954f5d73919,
                    0x11aefe640e277152,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf62f993117cf1dc5,
                    0xebd6cf8404109112,
                    0x3fe1b25e700f4193,
                    0x0ddcbb65068e3cda,
                ])),
                Felt::new(BigInteger256([
                    0xddee457e75576da8,
                    0x1508d7214145565e,
                    0x36536dde13a0f494,
                    0x08d996f8c99b6cc9,
                ])),
                Felt::new(BigInteger256([
                    0x2ace8f5e214f957b,
                    0x52b4a81ac77cb386,
                    0xc820efb202a2c614,
                    0x06dd4bc321fece42,
                ])),
                Felt::new(BigInteger256([
                    0xd0fc322dd73ba539,
                    0x2c5c8195625bf857,
                    0x23bcb4d09a265f53,
                    0x0a170d8438b645c5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xaea806217de338bb,
                    0x30a8703fa2bf64e8,
                    0xdf757ab7833a8b0f,
                    0x082fb6a46a6cf1fc,
                ])),
                Felt::new(BigInteger256([
                    0x5939efdc503d7dad,
                    0xe426680fab406a71,
                    0x0f0f628a68d2f46b,
                    0x052a54cc53a6b962,
                ])),
                Felt::new(BigInteger256([
                    0x779a28f44f95c36c,
                    0x811906e0605fd3c5,
                    0xe1759119e4dd84ca,
                    0x0978d1cb765e49f8,
                ])),
                Felt::new(BigInteger256([
                    0x23c78d50126c7e50,
                    0xd124e8181d5b0016,
                    0x228986eb888ff642,
                    0x01438aebefe3aaea,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8d0cfae160602c1e,
                    0x0426a6dad6583283,
                    0x48c9b858f2513d83,
                    0x03eec4651ff9a03b,
                ])),
                Felt::new(BigInteger256([
                    0xcc27b104f22106dd,
                    0x06942973593cb78b,
                    0x0c4f6d670c476bc1,
                    0x121eb2464cc62a6d,
                ])),
                Felt::new(BigInteger256([
                    0x53c1b8563c35ebe0,
                    0x2da03ed90e4a5689,
                    0x03bfd728c8573b7f,
                    0x05de98f53d1468b9,
                ])),
                Felt::new(BigInteger256([
                    0x514b88f54cd8de91,
                    0xb561007ff9b3c735,
                    0x3cdb205c5fe24479,
                    0x1048a60f6def596a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3cffb653920d3c89,
                    0x707332d53dd15da2,
                    0xba014ae8381a5f3b,
                    0x044ae98fd279b089,
                ])),
                Felt::new(BigInteger256([
                    0xa28dd04c769c3e09,
                    0xeba8c9901d04aed2,
                    0x4978c5ff4b542821,
                    0x0803dc8d4d666c18,
                ])),
                Felt::new(BigInteger256([
                    0x14c021a599c8229b,
                    0xd743c2be6564e937,
                    0xfeb9d715b949b12d,
                    0x0ffc409436e62822,
                ])),
                Felt::new(BigInteger256([
                    0x9e93493ff0dafc69,
                    0x8e2c72cf52aa1608,
                    0xdd3b13852dedbdcd,
                    0x0e99e4a9e453fcdc,
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
