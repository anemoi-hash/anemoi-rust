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
    let xy: [Felt; NUM_COLUMNS + 1] = [state[0], state[1]];

    let tmp = mul_by_generator(&xy[1]);
    state[0] = xy[0] + tmp;
    state[1] = mul_by_generator(&(tmp + xy[0])) + xy[1];
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
                    0xd5a3a0426e05ffa5,
                    0xa5c0e9b03b5a7e1a,
                    0x73157f691d4d3157,
                    0x11442391220ef410,
                ])),
                Felt::new(BigInteger256([
                    0x465e7074d75a5aaa,
                    0x05bf5deced0bce35,
                    0x29fceb0a8cdee652,
                    0x1f7c476691873268,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x93131ba8e928ce3b,
                    0x8be409f70dde9733,
                    0xaa71f8205ca0c84c,
                    0x1d947c182e799959,
                ])),
                Felt::new(BigInteger256([
                    0x37e8b8fa4ff03b6b,
                    0x9ec8ef97d469d865,
                    0x99f846fbb4481552,
                    0x09b3d5ce9136bfa1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x358216ce9e40f47b,
                    0xd2cfd0313bc7ddff,
                    0x1293cd59110d04cc,
                    0x3ed6e48bf5fbe051,
                ])),
                Felt::new(BigInteger256([
                    0x41e7f2914b3356b3,
                    0x2b34973ba3132094,
                    0x7207263eee3dfc1e,
                    0x04d6eb43de5ab83e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbd484ac236505a28,
                    0x85c580c4e0c874ce,
                    0xa2ec9eec211f6574,
                    0x4b5fdbf77176b1da,
                ])),
                Felt::new(BigInteger256([
                    0x8387e91a874321d5,
                    0xdcda8ebfed618ad0,
                    0xb579288ad609c80f,
                    0x4cb98d07a691b982,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xad3961b75017230c,
                    0x8f1d43ac8017ecce,
                    0xcdffbef591643643,
                    0x009216424356a200,
                ])),
                Felt::new(BigInteger256([
                    0x76f340bf5e1cd124,
                    0xe07cb9ec0b4fe094,
                    0x626856345ae8fe33,
                    0x3c772f5221bb060d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9596acc5a15944e4,
                    0xf6e1da94c37df3ca,
                    0xfc1a7d5b9cdfcbcb,
                    0x51aceb76e011ffec,
                ])),
                Felt::new(BigInteger256([
                    0x58a4d7af1f8cf3c6,
                    0x947e8d7688ad9998,
                    0x23f8c439370a8d5b,
                    0x6c93113011f1ecb1,
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
                    0xfffffffd00000003,
                    0xfb38ec08fffb13fc,
                    0x99ad88181ce5880f,
                    0x5bc8f5f97cd877d8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd8911e9adc2ec2d4,
                    0xcdbd79157c65fd99,
                    0xa5cb8b544c3fcd32,
                    0x11afd11d7283fd0f,
                ])),
                Felt::new(BigInteger256([
                    0xe0382dafc0f2eeca,
                    0x6146b70e5a1bb1c7,
                    0x459620b48f22f6ae,
                    0x0401d65ed2d49c7d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x771302c3f5618040,
                    0x44c87b26d94f61a6,
                    0x5df2507d818d7823,
                    0x247c577dd2968a65,
                ])),
                Felt::new(BigInteger256([
                    0xf6175f39e682cc4e,
                    0x6c9bae5ebdea9b7d,
                    0x1171dcdd97d4b28a,
                    0x15a15f7cda4f90c5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd24b91b9db2bd552,
                    0x3c90578db43c9f69,
                    0x4ef2e0b8926cb5b2,
                    0x6aa6a393a2ac4ff3,
                ])),
                Felt::new(BigInteger256([
                    0x43ada4a7cb55a1a4,
                    0x8c7455cc741aff26,
                    0x91a542035898e505,
                    0x477ac2682c617946,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xda57c3b5fd3ebfa8,
                    0xc271ec687bc167f1,
                    0x06f4a5e0045352af,
                    0x38c2376f1f1dbe4e,
                ])),
                Felt::new(BigInteger256([
                    0xa8befe744bcf5772,
                    0x752ea4e73177ffe0,
                    0xcd67eeeec7e3931b,
                    0x201f11428d31af50,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2202a93f6178a838,
                    0x30c211f5fc9f6101,
                    0xca2e036d85c24ce9,
                    0x351d4f32c3fc5a79,
                ])),
                Felt::new(BigInteger256([
                    0x2aca99701913444b,
                    0x73a6d321e7f3fc51,
                    0x782ba539b63ca96f,
                    0x469a43d4446a2dd9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x998d0fcc3c409fa6,
                    0x47ca120a82cb7057,
                    0x870f5baf8fb82d6e,
                    0x303f01324c134473,
                ])),
                Felt::new(BigInteger256([
                    0x2510bc4e6f4bae4c,
                    0xf6967463ffa5162c,
                    0x7597c0dd31d223b9,
                    0x6b5d7fdb3e845b3d,
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
                    0x9c180424823e88d4,
                    0xe28e6cff3543ad0d,
                    0x01060ad2711d7ed5,
                    0x5e6bfe84a37a9ea3,
                ])),
                Felt::new(BigInteger256([
                    0x822001b6cbc0fec1,
                    0xb8650e5606dd3825,
                    0x4dd85a7a27692b49,
                    0x4cec003daf47b411,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xea6b5a7f84e62b07,
                    0x7ca05ee9aa2a2452,
                    0x5e3c0742f253acc5,
                    0x0be2a403ecc6bcd7,
                ])),
                Felt::new(BigInteger256([
                    0xfdff7da95dfcb2c8,
                    0x369bc3673ac4d01d,
                    0x025e18276c237938,
                    0x278d006cb9b9d1b5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2488bc9bb33974bc,
                    0xf573928fbec52bc5,
                    0xa70e538d778e2c7a,
                    0x103bc4376fc3104a,
                ])),
                Felt::new(BigInteger256([
                    0x043cbe049d55c68b,
                    0x36e53b2a0379872e,
                    0xb58d7d8f55ae40cf,
                    0x5b24074c69a379b2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe718fa2a22dd7880,
                    0x95c3493bd500c52e,
                    0x340b9076f4f1f443,
                    0x68b5aef818d2e119,
                ])),
                Felt::new(BigInteger256([
                    0xe7946aef08af5bce,
                    0x6fb33141cf0d85f9,
                    0xe10090861c2db260,
                    0x5bc7c0a8d5a5648b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3bf993bccd145b9a,
                    0xf27517a5f04182c6,
                    0x07c8eda4f63d72f9,
                    0x2e7ce4e4e52570eb,
                ])),
                Felt::new(BigInteger256([
                    0xd8bc0ef4d4b2daf5,
                    0x59836adcbcebc2ed,
                    0x3044479730685317,
                    0x624882708aef25a9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8d8281770a1cbf3b,
                    0x26c32f5957f876a3,
                    0x3e9a6616eeb88531,
                    0x213d704d5212fe8d,
                ])),
                Felt::new(BigInteger256([
                    0xf1eac0e874811a1f,
                    0x239cc8f7ff294026,
                    0xe02097e437e8d0dc,
                    0x4318a6c7b0b03522,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x00000010ffffffef,
                    0x70681bcd001be411,
                    0x9928a7775c40a7a5,
                    0x4d37e37a3c8aae34,
                ])),
                Felt::new(BigInteger256([
                    0x0000007cffffff83,
                    0x1c66ea8900cd147d,
                    0xfcc184134bf98566,
                    0x64f54c64ae19d3be,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0000000efffffff1,
                    0x17e363d300189c0f,
                    0xff9c57876f8457b0,
                    0x351332208fc5a8c4,
                ])),
                Felt::new(BigInteger256([
                    0x0000006dffffff92,
                    0x048386b600b4786e,
                    0xfd252c8bdc752db6,
                    0x2fe21a441e542af9,
                ])),
            ],
            [
                Felt::one(),
                Felt::new(BigInteger256([
                    0x0000000efffffff1,
                    0x17e363d300189c0f,
                    0xff9c57876f8457b0,
                    0x351332208fc5a8c4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2af8102914858016,
                    0x4a9d9d4a65586a19,
                    0x21cf4c0154d475bf,
                    0x353bbb949e5d18b3,
                ])),
                Felt::new(BigInteger256([
                    0xaee872d95b677f58,
                    0xc77b6f55cc4d0ad8,
                    0xa0d5e66b5c52db74,
                    0x65c52b5486fae91e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdc67ca2316cf0e7d,
                    0x53676eb6458f1d26,
                    0x085b0046d4084d44,
                    0x38e25856ada07e3a,
                ])),
                Felt::new(BigInteger256([
                    0x04d604a1fda61830,
                    0x8336de5a21b38831,
                    0xa32d91ff1b780e06,
                    0x59f474d1fc44cd72,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4231eec10091e284,
                    0xd303fca6d720120c,
                    0x9dcb8a509f28ba0a,
                    0x4a93b2ae8327f1c4,
                ])),
                Felt::new(BigInteger256([
                    0xd39a4550a152f822,
                    0x594cefaae5623988,
                    0x05fd0d9b7fa21f00,
                    0x2189a5722fa7a3aa,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3c27e6b95fa8fb1c,
                    0xad37c9f67e69470a,
                    0x27b473f18066c4c6,
                    0x339a07a2f6a7b13b,
                ])),
                Felt::new(BigInteger256([
                    0x8cabba03a64e398f,
                    0x3100caf643f36344,
                    0x5d42340882178bbf,
                    0x693500241762c551,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x291dfc749df85847,
                    0x6e9b2b9d1abdaf4d,
                    0x264bd2970f4ca87d,
                    0x26e68a05b7fe88da,
                ])),
                Felt::new(BigInteger256([
                    0xf88df628267d44e3,
                    0x6488b01f78207a0c,
                    0xa2a981a07e9b6675,
                    0x16cd529f160c6bc7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2aedc7d639a47610,
                    0xd1161e15521fc7b8,
                    0x92972d344f8edb20,
                    0x27336276806e7d60,
                ])),
                Felt::new(BigInteger256([
                    0x1e6b37c60800548d,
                    0x33bc53873e0afe32,
                    0x7bcf2442518d1eb7,
                    0x6da5095ee07aa836,
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
