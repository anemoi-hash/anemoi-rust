use super::{sbox, BigInteger256, Felt};
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

/// Two elements (64-bytes) is returned as digest.
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
        let beta_y2 = y2 + y2.double().double();
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
        let beta_y2 = y2 + y2.double().double();
        *t += beta_y2 + sbox::DELTA;
    });

    state[..NUM_COLUMNS].copy_from_slice(&x);
    state[NUM_COLUMNS..].copy_from_slice(&y);
}

#[inline(always)]
/// Applies matrix-vector multiplication of the current
/// hash state with the Anemoi MDS matrix.
pub(crate) fn apply_mds(state: &mut [Felt; STATE_WIDTH]) {
    let xy: [Felt; NUM_COLUMNS + 1] = [state[0], state[1]];

    let tmp = xy[1] * mds::MDS[1];
    state[0] = xy[0] + tmp;
    state[1] = (tmp + xy[0]) * mds::MDS[1] + xy[1];
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
                    0xe6454136f82a54a2,
                    0xa20989feb614ac40,
                    0x17b0c8c87c9a2ccc,
                    0x2fa5ed3c638a68eb,
                ])),
                Felt::new(BigInteger256([
                    0xaab5fb2efc61aae6,
                    0x71b534112c919b2c,
                    0x96a585151a489f42,
                    0x3392ff78b8195126,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x09884f7cfba1b20c,
                    0x8895ea63a193f33b,
                    0x35ad01bf933d4ee8,
                    0x3b67f8b9c21a8168,
                ])),
                Felt::new(BigInteger256([
                    0xe3c017621264c384,
                    0xfd22a8d5720a7b3c,
                    0x348b65833a6a9e94,
                    0x1b4dbe155b4fbf29,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc6aff37526b674ae,
                    0xf6be0006c14295d2,
                    0xc5bd03c1ef89ff89,
                    0x0dc6c5f707300900,
                ])),
                Felt::new(BigInteger256([
                    0xf23c8ec3afca95df,
                    0x437d00cf1cbd5579,
                    0xbe58f874a98a083a,
                    0x119db69a3b9db78a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xda4fc836eef0667f,
                    0xee82e1003e034d1f,
                    0x64193e0100464bf6,
                    0x06e76b362c1b08ea,
                ])),
                Felt::new(BigInteger256([
                    0x7dcd2c2bc628d9e6,
                    0x848ed7a231810b3d,
                    0x7f92d0d4f277027b,
                    0x0318d7d9c85d3561,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe9518b8bb6f026fb,
                    0xb8c35ec74c2a4d31,
                    0x57dbf82a69da72ab,
                    0x3fe9927b3d8a7231,
                ])),
                Felt::new(BigInteger256([
                    0x0a916fb4a87ee746,
                    0x8e63287bcbd0c9cc,
                    0x38b6d0306f7c84fc,
                    0x2cd80ea04194f593,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6d17e22b0329a62d,
                    0x7b02fdfbe8ad2146,
                    0xc88d86bf1b4d760d,
                    0x1535280dd0085678,
                ])),
                Felt::new(BigInteger256([
                    0xe43745a183e61024,
                    0x8b6e931281b4f6fd,
                    0x7f2038bbad6aa987,
                    0x0a95c80d5b75a32d,
                ])),
            ],
        ];

        let output = [
            [
                Felt::new(BigInteger256([
                    0x123bd95299999999,
                    0xb83c0a9bfa40677b,
                    0xcccccccccccccccc,
                    0x0ccccccccccccccc,
                ])),
                Felt::zero(),
            ],
            [
                Felt::new(BigInteger256([
                    0xa942fcc6b6fc9019,
                    0xe30fee4001297b6b,
                    0xc81bbbb990652d5f,
                    0x151360a4ca1bade9,
                ])),
                Felt::new(BigInteger256([
                    0xc1f5f40ca5f0c922,
                    0x8ed096798f9a2565,
                    0x343bf2efffdc45e4,
                    0x136f8d515fc00212,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2daf6d68ef718427,
                    0x5061c937d5b3a764,
                    0xf8541bb755f5c730,
                    0x24e19c62db751b35,
                ])),
                Felt::new(BigInteger256([
                    0xffc6f91ea63fca57,
                    0x291c6ce7a87d6cc6,
                    0x0dde5b4ffa4ac0ea,
                    0x0d38cb1a91ae29c0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xeb95ce3a99999981,
                    0x819db2fb145092b5,
                    0xccccccccccccccc9,
                    0x0ccccccccccccccc,
                ])),
                Felt::new(BigInteger256([
                    0x311bac8400000004,
                    0x891a63f02652a376,
                    0x0000000000000000,
                    0x0000000000000000,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf2d9032a2d03b5eb,
                    0xcd2702a00e2c34db,
                    0x1603e7dc47c7c6c8,
                    0x0e59b0d670002df1,
                ])),
                Felt::new(BigInteger256([
                    0x5a31999fec1a5d35,
                    0x385e93baf78e0f5c,
                    0xe493dcf1046ad6d2,
                    0x1520885cd5546244,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x89714a402052a467,
                    0x7a619bb0c005c6ac,
                    0xb09715f3bf69fabb,
                    0x063449ebad11cda5,
                ])),
                Felt::new(BigInteger256([
                    0x69fb2e45307d0cce,
                    0xe11a60e5dcea0a32,
                    0xf5d12772ffc83700,
                    0x36cb03734b4cf542,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6a24fe5e1ab6a933,
                    0xec90d85b57906009,
                    0x3ee08fc1d7036de2,
                    0x0255c229795fa233,
                ])),
                Felt::new(BigInteger256([
                    0xf26c21fa41ab0838,
                    0x7f2614c13cbf784c,
                    0x4c8ba657fdaf62a2,
                    0x293a282cf7e6d932,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4b8a6b5ae7832826,
                    0x87665a8bd966d88c,
                    0x0aa7209abe900ad6,
                    0x159bbc937874d31d,
                ])),
                Felt::new(BigInteger256([
                    0x1e8d6f19ba3564f0,
                    0x392f1ff6ba684f78,
                    0x4a1b8c86d221cba0,
                    0x37727cf197730dfb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x04bd430a20b8705c,
                    0x32f05db3b070f7b8,
                    0x042d55ef902fb0c9,
                    0x0925894df8ff0e0d,
                ])),
                Felt::new(BigInteger256([
                    0x70eccdc8d4f0bfbc,
                    0x5c23c838081e47cf,
                    0x2a7ddc0abb283f89,
                    0x1a00118403cc6a54,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa46ad6157adab089,
                    0xa76e6a9e04acd6c7,
                    0x0777fc807bf591dc,
                    0x3129d3b76b000ed5,
                ])),
                Felt::new(BigInteger256([
                    0xb717d8913b050ed1,
                    0xd3286a7e0eb1c271,
                    0xd050a63db21a3cca,
                    0x376e2b55b36eb25d,
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
                    0xdf0a630fa75d2161,
                    0x18110560e32c261d,
                    0x0c0c980cded732a1,
                    0x1f5fd750ea47b21e,
                ])),
                Felt::new(BigInteger256([
                    0x80d77aa38fa77a06,
                    0xda886e9972b53d7e,
                    0x3a29b1dc37682474,
                    0x14d5886292f9be63,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa679006d62034ef4,
                    0x5401e755a84f7463,
                    0xbc863279e8730361,
                    0x0ff75a38d7ca1b22,
                ])),
                Felt::new(BigInteger256([
                    0x4b0b19418972d6ff,
                    0xe49aa60a6157c631,
                    0xa6ccf23db37b0a9a,
                    0x3146d838d2c1f45d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x60624907d17104ae,
                    0x600f01aa57ccad62,
                    0x3a24bdb04d2ea916,
                    0x19553be80a5d36c8,
                ])),
                Felt::new(BigInteger256([
                    0xbc97f61ed9b14f88,
                    0x1b0a2f162252e885,
                    0x89fb873771a89b9d,
                    0x23264485ede0ca6f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x28875f5685428fb1,
                    0x6c078f815d9529c7,
                    0x424e5422f3b8cfca,
                    0x3775743b87e16e8a,
                ])),
                Felt::new(BigInteger256([
                    0xb6ffa2f1bdcd5784,
                    0x6660470cb8b7708e,
                    0x5d97753703f2fe39,
                    0x31138c19b59e7ec8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x54dbdc0a1465efbd,
                    0xcaa47e1c4a8dc1d6,
                    0xc73ad3f63aab259f,
                    0x1f79b6576f3662c4,
                ])),
                Felt::new(BigInteger256([
                    0x45f63da62626037c,
                    0x1f2dc89eed3f1735,
                    0xb6eaf5fda123ab14,
                    0x0bce2dff622b5fd3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0095a40a4e0fa251,
                    0x2f58d2c4c719efbd,
                    0xd2a949a02cab98ff,
                    0x199e43c090c2c9fb,
                ])),
                Felt::new(BigInteger256([
                    0xb331538003e6b671,
                    0x0ff1e4b073b9193f,
                    0xb61c24422b5d6599,
                    0x1646d584b2346fbe,
                ])),
            ],
        ];

        let mut input2 = input;

        let output = [
            [Felt::zero(), Felt::zero()],
            [
                Felt::new(BigInteger256([
                    0x65a0e008ffffffe9,
                    0xeba8415b23a4d418,
                    0xfffffffffffffffc,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x99ed0724ffffff85,
                    0x88147ee76592dd8d,
                    0xffffffffffffffef,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x96bc8c8cffffffed,
                    0x74c2a54b49f7778e,
                    0xfffffffffffffffd,
                    0x3fffffffffffffff,
                ])),
                Felt::new(BigInteger256([
                    0x8f7765b8ffffff99,
                    0x3598729825300edc,
                    0xfffffffffffffff2,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::one(),
                Felt::new(BigInteger256([
                    0x96bc8c8cffffffed,
                    0x74c2a54b49f7778e,
                    0xfffffffffffffffd,
                    0x3fffffffffffffff,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4ab1f1ff75a2837d,
                    0x182dfc680d8d07db,
                    0x2edd1159f3dfe8e9,
                    0x078b813dc9286a0e,
                ])),
                Felt::new(BigInteger256([
                    0xf65134a0dbd40b77,
                    0x536e5ca1b67664c6,
                    0x247b089dfac7b102,
                    0x3a8f0e9780c3d0aa,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xec94d231114181eb,
                    0x41ecc19968b3afe3,
                    0xfe86edae69da3867,
                    0x06599354f593e0f6,
                ])),
                Felt::new(BigInteger256([
                    0x5dac4915dfba6095,
                    0x0bf3d50d63458cc7,
                    0x9f6f96a5c4be249f,
                    0x1106b8e19ea55930,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6a85563f11e79253,
                    0x806e2224e6ad3d66,
                    0xec0e61c58579b327,
                    0x09149285afc12af5,
                ])),
                Felt::new(BigInteger256([
                    0x44ebba3933372b26,
                    0x7aea40d29a2072a8,
                    0x264370130d091b62,
                    0x108d21225ca6a13d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8a69e18b3a454541,
                    0xe2ce8ed0d2d7b91a,
                    0x16439e360777c6e8,
                    0x2cd730bc13f9e874,
                ])),
                Felt::new(BigInteger256([
                    0x39f55e25e127b1c5,
                    0x4b4ead30b09b6a9d,
                    0xcce98c452949e0c5,
                    0x11477fc61980090c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x26642527d3240128,
                    0x4442d03ae3348d03,
                    0x59d1a1ea605d7d04,
                    0x1a809c545a0f41e7,
                ])),
                Felt::new(BigInteger256([
                    0xed5d212b45da0942,
                    0x2feea7cd4a1c8689,
                    0x78031f9182f71c29,
                    0x10513ba52477a958,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x67fe6f4861913284,
                    0x3a85183ef68e1c40,
                    0x6135feeb057e94fc,
                    0x09006f580bc8f8b5,
                ])),
                Felt::new(BigInteger256([
                    0x2ee294c8ebbcb304,
                    0x1244c4ef3aeafda4,
                    0x9c2a1ed946d64e86,
                    0x0349023ced214b49,
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
