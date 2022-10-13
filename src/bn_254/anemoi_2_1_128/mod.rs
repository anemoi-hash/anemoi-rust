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
        let beta_y2 = y2.double() + y2;
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
        let beta_y2 = y2.double() + y2;
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
                *r += *s * mds::MDS[i * STATE_WIDTH + j];
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
                Felt::new(BigInteger256([
                    0x90faad8946047d7c,
                    0x89118168ec518379,
                    0x8692cc020e6212a8,
                    0x0af5c943ca1bff16,
                ])),
                Felt::new(BigInteger256([
                    0xb642c0062e7601b5,
                    0xfe645fba39cee1a0,
                    0xa10cadd382d2c569,
                    0x25e6c8feeae40fc1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe6c7b8707a0bb2e7,
                    0xb2e705f872b39f93,
                    0xb976e674b6cf29f7,
                    0x0c32c47562708241,
                ])),
                Felt::new(BigInteger256([
                    0xec8741300f7f1644,
                    0xfe64a10a84aabc5d,
                    0x9192a3e320590a6a,
                    0x27ebe18205bae2a6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb31ddfc411b6e3b9,
                    0xb5891a3abc6892db,
                    0xe6fdb7a9432c5da0,
                    0x28b8e4675f1b57a7,
                ])),
                Felt::new(BigInteger256([
                    0x0b28b1cb2b666acb,
                    0xf94c5d54f5b7faa1,
                    0x9632cc0db40469a3,
                    0x0404b2783446c352,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0432000d61be4685,
                    0xdefc4510e33e5c7f,
                    0x17e992246440ca1b,
                    0x21acd491a8456ed9,
                ])),
                Felt::new(BigInteger256([
                    0xacdc86db7edea1a4,
                    0x7c19542a7e3a5e52,
                    0xf91f1dd12bb2911c,
                    0x0b0c30f7d4a5efae,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x09717504eefd9d77,
                    0x87407661b144dae8,
                    0xd52e90215018386b,
                    0x19ecd2685ef0a66c,
                ])),
                Felt::new(BigInteger256([
                    0xfac92ada47c2a9d2,
                    0xb8bb2aae14e169d5,
                    0xd9f7678faf8745e7,
                    0x17c49b7e1469658b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3a389b1decff38ff,
                    0xf207678420391cbc,
                    0x07ccc13e093ece6c,
                    0x28a06d5c76a545d2,
                ])),
                Felt::new(BigInteger256([
                    0xc832849fc12f8737,
                    0x4cf831f360b6460c,
                    0xe75f08b7e146b683,
                    0x28354e975bf75981,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(), Felt::zero()],
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
            ],
            [
                Felt::new(BigInteger256([
                    0x7a17caa950ad28d7,
                    0x1f6ac17ae15521b9,
                    0x334bea4e696bd284,
                    0x2a1f6744ce179d8e,
                ])),
                Felt::new(BigInteger256([
                    0xc9638b5c069c8d94,
                    0x39b65a76c8e2db4f,
                    0x8fb1d6edb1ba0cfd,
                    0x2ba010aa41eb7786,
                ])),
            ],
            [
                Felt::one(),
                Felt::new(BigInteger256([
                    0x7a17caa950ad28d7,
                    0x1f6ac17ae15521b9,
                    0x334bea4e696bd284,
                    0x2a1f6744ce179d8e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3b81d56e206c880d,
                    0x553bcb74c8da9341,
                    0xf9184a0f93d7b22b,
                    0x1be1875ac864ee07,
                ])),
                Felt::new(BigInteger256([
                    0xf0872822dec19f4e,
                    0xcf14ecf5c37b0649,
                    0x1bb500953b572b30,
                    0x18c2c22981af9986,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x341c63d2f78efb25,
                    0x7f1213f52fd03f93,
                    0xfd8e46b114d7987d,
                    0x232dcc15b13de9e1,
                ])),
                Felt::new(BigInteger256([
                    0xd47ac8646cb50fde,
                    0xb5169d35dac61b6f,
                    0x614ca6d2da5bcaca,
                    0x00485a6a75dfbfcf,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9877690ebb6d26d3,
                    0x09ecc7a8351eb831,
                    0xf145d61bddb8422f,
                    0x0462ad5d1abe0175,
                ])),
                Felt::new(BigInteger256([
                    0xd48eecf75daddf44,
                    0x1712b44d95142335,
                    0x6a044e614d2d3031,
                    0x112cba8f8480c7b4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xcea7088905dd2e2a,
                    0xbbc6d6fef57bace9,
                    0x4af6a5e165d72513,
                    0x126d190645059dbc,
                ])),
                Felt::new(BigInteger256([
                    0xdcb1145fb7f92edb,
                    0x17ec6e95f63b9a82,
                    0x21b2c9bedbb6a7fa,
                    0x11ef2d97c28528ba,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x818bdd66154ba05f,
                    0x826f21491f05834f,
                    0xf2743b635bab5967,
                    0x007207fcd9c996bc,
                ])),
                Felt::new(BigInteger256([
                    0x7f6cc30c87a58aef,
                    0x40088e8971f1f3c4,
                    0xb15419b9c289521e,
                    0x191ab374a1c629c2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xde6e84b8a716d6cf,
                    0x126bbdaa09068f3a,
                    0x94f90a42288ee8de,
                    0x10136dc9e6f671da,
                ])),
                Felt::new(BigInteger256([
                    0x275d86b2ddf70e5d,
                    0xecba006013582930,
                    0xedf9e1c7d97218bf,
                    0x280b49822fa90ee7,
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
