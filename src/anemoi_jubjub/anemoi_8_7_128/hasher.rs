//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{apply_permutation, DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
use super::{Jive, Sponge};

use super::Felt;
use super::{One, Zero};

use ark_ff::FromBytes;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// An Anemoi hash instantiation
pub struct AnemoiHash {
    state: [Felt; STATE_WIDTH],
    idx: usize,
}

impl Default for AnemoiHash {
    fn default() -> Self {
        Self {
            state: [Felt::zero(); STATE_WIDTH],
            idx: 0,
        }
    }
}

impl Sponge<Felt> for AnemoiHash {
    type Digest = AnemoiDigest;

    fn hash(bytes: &[u8]) -> Self::Digest {
        // Compute the number of field elements required to represent this
        // sequence of bytes.
        let num_elements = if bytes.len() % 31 == 0 {
            bytes.len() / 31
        } else {
            bytes.len() / 31 + 1
        };

        let sigma = if num_elements % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 31-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut i = 0;
        let mut num_hashed = 0;
        let mut buf = [0u8; 32];
        for chunk in bytes.chunks(31) {
            if num_hashed + i < num_elements - 1 {
                buf[..31].copy_from_slice(chunk);
            } else {
                // The last chunk may be smaller than the others, which requires a special handling.
                // In this case, we also append a byte set to 1 to the end of the string, padding the
                // sequence in a way that adding additional trailing zeros will yield a different hash.
                let chunk_len = chunk.len();
                buf = [0u8; 32];
                buf[..chunk_len].copy_from_slice(chunk);
                // [Different to paper]: We pad the last chunk with 1 to prevent length extension attack.
                if chunk_len < 31 {
                    buf[chunk_len] = 1;
                }
            }

            // Convert the bytes into a field element and absorb it into the rate portion of the
            // state. An Anemoi permutation is applied to the internal state if all the the rate
            // registers have been filled with additional values. We then reset the insertion index.
            state[i] += Felt::read(&buf[..]).unwrap();
            i += 1;
            if i % RATE_WIDTH == 0 {
                apply_permutation(&mut state);
                i = 0;
                num_hashed += RATE_WIDTH;
            }
        }

        // We then add sigma to the last register of the capacity.
        state[STATE_WIDTH - 1] += sigma;

        // If the message length is not a multiple of RATE_WIDTH, we append 1 to the rate cell
        // next to the one where we previously appended the last message element. This is
        // guaranted to be in the rate registers (i.e. to not require an extra permutation before
        // adding this constant) if sigma is equal to zero. We then apply a final Anemoi permutation
        // to the whole state.
        if sigma.is_zero() {
            state[i] += Felt::one();
            apply_permutation(&mut state);
        }

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        AnemoiDigest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn hash_field(elems: &[Felt]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        let sigma = if elems.len() % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        let mut i = 0;
        for &element in elems.iter() {
            state[i] += element;
            i += 1;
            if i % RATE_WIDTH == 0 {
                apply_permutation(&mut state);
                i = 0;
            }
        }

        // We then add sigma to the last register of the capacity.
        state[STATE_WIDTH - 1] += sigma;

        // If the message length is not a multiple of RATE_WIDTH, we append 1 to the rate cell
        // next to the one where we previously appended the last message element. This is
        // guaranted to be in the rate registers (i.e. to not require an extra permutation before
        // adding this constant) if sigma is equal to zero. We then apply a final Anemoi permutation
        // to the whole state.
        if sigma.is_zero() {
            state[i] += Felt::one();
            apply_permutation(&mut state);
        }

        // Squeezing phase

        AnemoiDigest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiHash {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.try_into().unwrap();
        apply_permutation(&mut state);

        let mut result = [Felt::zero(); NUM_COLUMNS];
        for (i, r) in result.iter_mut().enumerate() {
            *r = elems[i] + elems[i + NUM_COLUMNS] + state[i] + state[i + NUM_COLUMNS];
        }

        result.to_vec()
    }

    fn compress_k(elems: &[Felt], k: usize) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);
        assert!(STATE_WIDTH % k == 0);
        // We can output as few as 1 element while
        // maintaining the targeted security level.
        assert!(k <= STATE_WIDTH);

        let mut state = elems.try_into().unwrap();
        apply_permutation(&mut state);

        let mut result = vec![Felt::zero(); STATE_WIDTH / k];
        let c = result.len();
        for (i, r) in result.iter_mut().enumerate() {
            for j in 0..k {
                *r += elems[i + c * j] + state[i + c * j];
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::super::BigInteger256;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger256([
                0x098ef3aec5ad586e,
                0x69aea51f3b560a8d,
                0x2d9c74c6aa8d5f74,
                0x11f7a85e44f67c58,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x40379487020d35c8,
                    0x2730cf2da04a98ed,
                    0x413ff910cd505856,
                    0x30df0381c3245cb6,
                ])),
                Felt::new(BigInteger256([
                    0x0c24f30273c11688,
                    0x86e13aa9f93d0c85,
                    0xe86790eb3eb9e2e3,
                    0x3018610afced81e1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x19577d508d0a8eaf,
                    0xa0c86165ed0c3e72,
                    0x8097ddb02fe87c52,
                    0x1a235c80b7ec2728,
                ])),
                Felt::new(BigInteger256([
                    0xdf42ad469d7b3d22,
                    0x3acccbbab95fab5e,
                    0x2b10daf04dd061fd,
                    0x492f8c5aeae48565,
                ])),
                Felt::new(BigInteger256([
                    0x8883b26331108036,
                    0x3df747b9d84446c1,
                    0xb147544a2e7fb3eb,
                    0x153167166bd9de45,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x39bd598d906d9f2c,
                    0x9c6d71b094726070,
                    0xcfee8ad5abfc0de3,
                    0x2fb92af33be8b55d,
                ])),
                Felt::new(BigInteger256([
                    0x6ac2d7302f40145e,
                    0x5d5f989f2bc0aae3,
                    0x2ee9513c06b7a361,
                    0x4d689d1857bb9465,
                ])),
                Felt::new(BigInteger256([
                    0xb2a06c849ddcf25a,
                    0x64e3065f66eb1d11,
                    0xb128b5590080a2c0,
                    0x1c030790e33ae4e6,
                ])),
                Felt::new(BigInteger256([
                    0x14a7376d53afc1d7,
                    0xe3c761311e2ebc6c,
                    0x74e94c877c8768b6,
                    0x3a25669e64e7890a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7ad40afa222950e4,
                    0xe97ad9c8d84fd91d,
                    0x825d2d4cb8cfeb5e,
                    0x531e1d0b28f7e6cc,
                ])),
                Felt::new(BigInteger256([
                    0x5d3afd04c6c9ca45,
                    0x87f18b88aac04ca4,
                    0xc40ab76b023a5abb,
                    0x288e0a83cd2ad582,
                ])),
                Felt::new(BigInteger256([
                    0xeb196d6e9abef7f6,
                    0x1e36a3ba46a7d965,
                    0xcf5f5e452d4d6252,
                    0x70c4dafd772998a4,
                ])),
                Felt::new(BigInteger256([
                    0xb229a88271482fbf,
                    0x00de0aef4ddadf59,
                    0xa98550f4ee2963f4,
                    0x5c88318d67ef41aa,
                ])),
                Felt::new(BigInteger256([
                    0xcb9ce63941f2c7e9,
                    0x232b7b60055e9dd0,
                    0xad5f96bfe58dfcbe,
                    0x0fb7a266e835d095,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc618bb845bca8689,
                    0xa8d16d9aa742f12a,
                    0x10431f1bf0d76cc6,
                    0x1e7b8bc96593a00c,
                ])),
                Felt::new(BigInteger256([
                    0x34a4df8f785c7f66,
                    0x5b9eeeb7f0fd4c31,
                    0xf9d5f8d3d1524bd6,
                    0x57e237f2d670ac2a,
                ])),
                Felt::new(BigInteger256([
                    0xc443bbe49efedee4,
                    0x57b1350126edc029,
                    0x83a2adeacac34c60,
                    0x348353bb26f51c05,
                ])),
                Felt::new(BigInteger256([
                    0x9354f4e7ad52d8f0,
                    0x2cd47cb3fe4af2df,
                    0x4c40e0074123819a,
                    0x661ed5ab235dc340,
                ])),
                Felt::new(BigInteger256([
                    0xa2b21b8d754b13c3,
                    0xa519e8fca427ea95,
                    0x299140180296bc00,
                    0x67ba3f8dbd2aa23b,
                ])),
                Felt::new(BigInteger256([
                    0x756c05409d7c8e31,
                    0xa721e4244d8e2592,
                    0xddbbcdfd4c609bbf,
                    0x6b29ff21176a9e1e,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x3380460612f5a125,
                0xf8132967de120352,
                0x0a6b9426d74ffc39,
                0x0d9260d1e441b9a4,
            ]))],
            [Felt::new(BigInteger256([
                0x53f1e864e3f05d79,
                0x189efabb713af133,
                0x947827c237d1d557,
                0x5552300d8c43728e,
            ]))],
            [Felt::new(BigInteger256([
                0x463b45607c3e9aff,
                0xe5764d634fc72492,
                0x99df2ca67a932d7a,
                0x485580ab6539d578,
            ]))],
            [Felt::new(BigInteger256([
                0x557b588ef8d2e52e,
                0xc01844b4eb383fea,
                0x97ec1a66fcd1128c,
                0x6222d129d43f8ecc,
            ]))],
            [Felt::new(BigInteger256([
                0xd8b3c49689f12d6e,
                0x2b48a7c4db1aedd1,
                0x3809d7ee289db94a,
                0x3e7f3a855fd0e617,
            ]))],
            [Felt::new(BigInteger256([
                0x44d09056ce4aedd6,
                0x272465a21c9af7ed,
                0xd783000a7e980212,
                0x43ff2d5f4c984d17,
            ]))],
            [Felt::new(BigInteger256([
                0x066c6bf31edb245d,
                0x27d6d3a210d79a6a,
                0x585487226dc91fff,
                0x1713cc1a36856086,
            ]))],
            [Felt::new(BigInteger256([
                0xa05bd4044682b26b,
                0xa8b7c27c11ca3ef0,
                0xa316f9b5ea327d0d,
                0x4769ed0eec898611,
            ]))],
            [Felt::new(BigInteger256([
                0xb9935e26c2884add,
                0x694573dff55acd5e,
                0xe62d54bcb019481e,
                0x432b2dd92addc604,
            ]))],
            [Felt::new(BigInteger256([
                0x6d3d4f17b93b36ef,
                0xb9300bd94ef84843,
                0x1688be54d9f45c84,
                0x6cbb4f92948b9023,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 8],
            vec![Felt::one(); 8],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x503a6e8484257256,
                    0x3e34a1b5bfc40ff8,
                    0x2a50711dcdb89988,
                    0x6d51a2136640018d,
                ])),
                Felt::new(BigInteger256([
                    0xe7d30ed6e47c663b,
                    0x27bedaf349d8bc26,
                    0xeb2a9a9de23e5fc9,
                    0x19d63baa35311122,
                ])),
                Felt::new(BigInteger256([
                    0x7ad81df87371c9e6,
                    0x308ddd7f0f6943a7,
                    0x4b9d9494bdc3efc3,
                    0x2cc4a085fd334b05,
                ])),
                Felt::new(BigInteger256([
                    0xc322a8f91dfdb8c8,
                    0xc2578c1297551232,
                    0xcaef56d7725502c9,
                    0x11ffd74266296e73,
                ])),
                Felt::new(BigInteger256([
                    0x63e68d45062246d2,
                    0x325c9c0544f6120b,
                    0x27821dcade91e4d0,
                    0x0b5bd655fd54533e,
                ])),
                Felt::new(BigInteger256([
                    0xcb0c2879f90c0a27,
                    0xea390772b99364ca,
                    0x8618f9bfeab9977a,
                    0x087365a0953da0c5,
                ])),
                Felt::new(BigInteger256([
                    0xd520c3edff843344,
                    0xddafd673d3ac4731,
                    0xc8837cf615c04a4c,
                    0x41c14940e54d7c9c,
                ])),
                Felt::new(BigInteger256([
                    0xdf099dfdfa370b48,
                    0x9ce7d9fda5c19a28,
                    0x8c45eb4b29666fe2,
                    0x2fa6752c650da778,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x4405d6e083a8f7ee,
                    0xe0c10a82f85d8c7c,
                    0xe36ef18ec7efa8e3,
                    0x5b43f2e7d4309a7f,
                ])),
                Felt::new(BigInteger256([
                    0x80b9d9e0d59fedee,
                    0xd4d5c7bcdba8fd02,
                    0x75ee214d69cf1500,
                    0x06f9f7b13058e63c,
                ])),
                Felt::new(BigInteger256([
                    0x8a56eda5531288a6,
                    0x2640f87b016f1ac4,
                    0xc16e0720602cb7f2,
                    0x6ab50b2cd8891ed2,
                ])),
                Felt::new(BigInteger256([
                    0xfc5b11c593798b89,
                    0xf50d30728e47d295,
                    0xe74f7b641845ce87,
                    0x1e3409ad7fdcc1e4,
                ])),
                Felt::new(BigInteger256([
                    0x99a99e6f1b24a6c3,
                    0xd350f157a400611a,
                    0xde120df9a97517cd,
                    0x1e34388d7a142de4,
                ])),
                Felt::new(BigInteger256([
                    0xeeaf8bf2697c9e12,
                    0x9d78263c9b993a39,
                    0xbaad3c28fcc0e2da,
                    0x10944e0cb68ac8ab,
                ])),
                Felt::new(BigInteger256([
                    0x90e5183d707be70e,
                    0x4482a975dce63a73,
                    0xe4b0b90e0d666a04,
                    0x65015535003203ee,
                ])),
                Felt::new(BigInteger256([
                    0x6d73fbafdd69d20c,
                    0xfec11a51835430a1,
                    0xe05353199104397b,
                    0x65dcce2b926a18fa,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xdbd56ff263857941,
                    0x14d30b5579233914,
                    0x2525f93a9ee729fe,
                    0x5584babc316d6273,
                ])),
                Felt::new(BigInteger256([
                    0x2897283b504ba47d,
                    0x276cde239f9e8f68,
                    0x674a037210d392b1,
                    0x4bad0286c7f4dd15,
                ])),
                Felt::new(BigInteger256([
                    0xe5b12d8debe538ef,
                    0xdeb1756e8723766c,
                    0x5c313c6fcfa51cca,
                    0x6dc2313a8cf9562e,
                ])),
                Felt::new(BigInteger256([
                    0x53ded7a76ccdd8fd,
                    0x714ef7f0b150a421,
                    0xf3c49ca220801196,
                    0x3e9645004519a1e6,
                ])),
                Felt::new(BigInteger256([
                    0x3693d7af52218be4,
                    0x380f898a4090408c,
                    0x4a8ea07c9768e2ce,
                    0x69b19a98caa94e0c,
                ])),
                Felt::new(BigInteger256([
                    0xa9fb475a21479954,
                    0x5ccd0f184782a654,
                    0x6979362203d75e13,
                    0x26587594efd9032f,
                ])),
                Felt::new(BigInteger256([
                    0xce8b7f359232e5b8,
                    0x9386d3a4a631df8d,
                    0x2025d1e143a9f637,
                    0x0d44f56fb5bacfab,
                ])),
                Felt::new(BigInteger256([
                    0x7fcd73d9871acf6a,
                    0x0038a05425a6c07b,
                    0x4376503d3670294d,
                    0x27cc31593b22ac35,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5e95c676ffab396b,
                    0x4294680c53430656,
                    0x3bb2dcbafd3b597b,
                    0x3ddfcfe5377352c9,
                ])),
                Felt::new(BigInteger256([
                    0x86f51bb0ebf82538,
                    0xdd65c0272adda95c,
                    0xf8193a5964514ee3,
                    0x5ad562600a1768df,
                ])),
                Felt::new(BigInteger256([
                    0xd1d78d6b4a41a49b,
                    0x12bf7039e3076cd0,
                    0x44c88c41342980db,
                    0x484da8b0f5dc50d3,
                ])),
                Felt::new(BigInteger256([
                    0xcf7e4bf4780d4367,
                    0x3f72f54d3de2cb8b,
                    0x8db52c08cf912aa9,
                    0x6729e44edaeea7fe,
                ])),
                Felt::new(BigInteger256([
                    0xc1df76540efd5506,
                    0xd2bd6ef707b7218d,
                    0x229c50f3dc01cc33,
                    0x59b459d8bedc4392,
                ])),
                Felt::new(BigInteger256([
                    0x9a5ca27f50c9b2b2,
                    0xf9e31b27fe591141,
                    0xb733eb40b3081d17,
                    0x294414dfe19ac85c,
                ])),
                Felt::new(BigInteger256([
                    0x2de3635c127a353b,
                    0x5c6b8183ccd2d32f,
                    0xabb42c6a3309db14,
                    0x68cf78730b75debb,
                ])),
                Felt::new(BigInteger256([
                    0x30cf2c45300b9492,
                    0xbb2dbf77438bd37f,
                    0x542f52c639644c35,
                    0x1e47c8cb9449d4af,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x8b8ab088b382eb43,
                    0xbf6740b0739440c4,
                    0xda9dce840be2275a,
                    0x6d651ee90a63fbbe,
                ])),
                Felt::new(BigInteger256([
                    0x74737266e6651e59,
                    0x3b5a76808635da87,
                    0x8da98e10f8034f6c,
                    0x53121db329b0f754,
                ])),
                Felt::new(BigInteger256([
                    0x3a819c148d9072a1,
                    0x8a2cd1044afac626,
                    0x3d24925618403cf2,
                    0x62cdc9e852ad11f2,
                ])),
                Felt::new(BigInteger256([
                    0x422287f9d0ce3457,
                    0x700819f4b90f3a6f,
                    0xddcf2504e73ec2e9,
                    0x0ba23439c831fd3c,
                ])),
                Felt::new(BigInteger256([
                    0xbbe4d81f50f65722,
                    0x788207c751a4f07b,
                    0x0cfc4f1c49a12619,
                    0x45199fd5e578eb04,
                ])),
                Felt::new(BigInteger256([
                    0x2c8c15df0621dda6,
                    0x614da493ca7b65ef,
                    0x2877bd7e4abe5c11,
                    0x02a197b3768c04b8,
                ])),
                Felt::new(BigInteger256([
                    0x581f7e5343e2ec81,
                    0x3436b478579013a8,
                    0x27374a6f07361054,
                    0x04a0d8940e6d7018,
                ])),
                Felt::new(BigInteger256([
                    0xfdef4cf7ddef3ae2,
                    0xfa8a830f86f1b21d,
                    0x60b3bfcc6c9c481d,
                    0x2fef4a3c2c0c23be,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xfe77a8a2d32bb691,
                    0xee1490994a72c0f0,
                    0x7287f865b37d6185,
                    0x32ad53e4dc801232,
                ])),
                Felt::new(BigInteger256([
                    0x23affda2804d0480,
                    0xadd8c676eedf3cdf,
                    0xe661b1ddd9e92a63,
                    0x55ffb78f1defb743,
                ])),
                Felt::new(BigInteger256([
                    0x25aa6a95fe0f5221,
                    0x03ea27ceee9fd149,
                    0x1a338945d75936c6,
                    0x603bc9f07f32fec2,
                ])),
                Felt::new(BigInteger256([
                    0x8612ff20f2cfd64e,
                    0x3fcbefe65fb3fdf1,
                    0xa2a49c31e42b2338,
                    0x26b9040cc41f4094,
                ])),
                Felt::new(BigInteger256([
                    0x6d55b84ca0cfa314,
                    0x59b58e9a4bb72e4f,
                    0x234cf8e8f9799470,
                    0x40f4bdb4e196a43c,
                ])),
                Felt::new(BigInteger256([
                    0x5d10e1687ee7d877,
                    0x520f0594bb944b53,
                    0x7041acbaf6aacd25,
                    0x53098a0c89e39c24,
                ])),
                Felt::new(BigInteger256([
                    0xe0b6b7cdd7a2714e,
                    0xe2b13bbb9653f451,
                    0x6fe9053bf617de00,
                    0x3ff51dcd172b2524,
                ])),
                Felt::new(BigInteger256([
                    0xe7f9d8ee329eee1d,
                    0xa146d4de4b85f3f0,
                    0x3ae59a5b686663ca,
                    0x0f8bfd387c58ca78,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x456d39e02c8085cc,
                    0xa99038707db2066d,
                    0x00b79652f1b6afbd,
                    0x48a61cc3547602c1,
                ])),
                Felt::new(BigInteger256([
                    0xeee936c7aad0369f,
                    0x392690e2753b76b2,
                    0x6335f8d753a49d7f,
                    0x08f8ca00f38b4131,
                ])),
                Felt::new(BigInteger256([
                    0xd9c7401c02e564e9,
                    0xa093ccfe40c1a76c,
                    0xd4bb24365ad1fef7,
                    0x51745b32e34d16d7,
                ])),
                Felt::new(BigInteger256([
                    0x16fa4be96d3f6cca,
                    0x31d88ccb2e69edd6,
                    0x7c4ac6444b9549d6,
                    0x4661630431b12984,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x89e0e30cbe651e7f,
                    0x9ad7c730bd75d24c,
                    0xd8270cd80f79e600,
                    0x41b60f914068aede,
                ])),
                Felt::new(BigInteger256([
                    0xcdd0df6bb9992c84,
                    0xa6229e3f5c753a3f,
                    0xd6316da43f7cb11a,
                    0x49737c5e06e8844f,
                ])),
                Felt::new(BigInteger256([
                    0x4ac8ff1f850a747e,
                    0x4528a64bf27e13f4,
                    0xb82b14991b04b633,
                    0x500392f6bb70eb86,
                ])),
                Felt::new(BigInteger256([
                    0x9f309047817bf5e3,
                    0x9c510401d4fbe408,
                    0x67a6afc396198ec4,
                    0x1355ee0b2ca32ae9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa2bc90152e6ca7c0,
                    0xd1f8a7472bdcd4ca,
                    0x286112fecaeca77f,
                    0x1ef6fb5af89fce70,
                ])),
                Felt::new(BigInteger256([
                    0x329a2e11882a7785,
                    0x99f3f8593fbd870f,
                    0xdefd6bd0ee73bdee,
                    0x2b03b7dcc6e1d45a,
                ])),
                Felt::new(BigInteger256([
                    0xa8e1ce59322d577a,
                    0x9739a4bcb66512de,
                    0x91a0bd7578ba784b,
                    0x32a1993d6e9f2bb1,
                ])),
                Felt::new(BigInteger256([
                    0x302d72b381adfac1,
                    0xc9accdf4abdd6f66,
                    0xd6d0546bca90263c,
                    0x1f1a8d74579dffab,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe536e1540dea6395,
                    0xbddc315668a6d433,
                    0xbc220e324b7f55d5,
                    0x59ee7babf662e00e,
                ])),
                Felt::new(BigInteger256([
                    0xd851df0aaa3eefce,
                    0xa622c3a7ad331415,
                    0xec58b16b525b7154,
                    0x730cdc1399ce768d,
                ])),
                Felt::new(BigInteger256([
                    0x80d66d6d3696403d,
                    0xc07e530cc6ecb356,
                    0xd23e079f7c97fe98,
                    0x237e09c1a7960ab4,
                ])),
                Felt::new(BigInteger256([
                    0x9e99dd7947af59d0,
                    0x29e2adcd99938307,
                    0x073f32e2519dcf0e,
                    0x5faf8604828e1b33,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x563e3a08467375a9,
                    0x3b40ca52c0835482,
                    0x0fbea4cdb478e81a,
                    0x30a5a1b4e328fbb1,
                ])),
                Felt::new(BigInteger256([
                    0x9a673b5b40f37cbe,
                    0x6e138f8eea1f0f7f,
                    0x9aa821339757d3b1,
                    0x04e8009fe85398cb,
                ])),
                Felt::new(BigInteger256([
                    0xce29a28efaa0e894,
                    0xfe65058d7f3908bf,
                    0xae74504cc5e017e7,
                    0x3e79b71050cfe686,
                ])),
                Felt::new(BigInteger256([
                    0xa2893c598e71abb1,
                    0xcc076e03292012ad,
                    0xf11932201981dee6,
                    0x572570458fa6b0be,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x543f67c301303d26,
                    0xbf8165ad6c023635,
                    0xd4fb0acc93a11743,
                    0x60aff9d936edd57f,
                ])),
                Felt::new(BigInteger256([
                    0x448e35fba0180815,
                    0x478ddf97b595ca52,
                    0xe1252599e6cbd3da,
                    0x36ae9cdbacc951e2,
                ])),
                Felt::new(BigInteger256([
                    0x045404ba9600a3fc,
                    0x3a363b883ac0794d,
                    0x01750f6da7e6ee2b,
                    0x43b4e2794c969bb1,
                ])),
                Felt::new(BigInteger256([
                    0x22d1749420690515,
                    0xc47edf2da3c0d1d2,
                    0x9436582af78fec00,
                    0x0c53bded773e4823,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe4efc44bc7e21e34,
                    0x64d84b50ebb03292,
                    0xe184b43c4c2d02fd,
                    0x5dc655de0c941abe,
                ])),
                Felt::new(BigInteger256([
                    0x05a7d74247c9b3b9,
                    0x943fd396a7e7938d,
                    0x5ab94b754950642a,
                    0x574bef42e55ab2fb,
                ])),
                Felt::new(BigInteger256([
                    0x72498b5c51f2ad4f,
                    0x1ed661cbbb0ba07c,
                    0x8fdc21ed5eac82ca,
                    0x0fdf7da4b216c959,
                ])),
                Felt::new(BigInteger256([
                    0x7debb110ca431110,
                    0x2d20f24d1e98f070,
                    0xdc025a9479d22d65,
                    0x3ca8634225c6aa8a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x39ab51e24ed78ae1,
                    0x9723884b5554b033,
                    0x3b038535e17ee713,
                    0x04df2aab15b9a3a6,
                ])),
                Felt::new(BigInteger256([
                    0x8d6ae85806497cb6,
                    0x188266089a4ae3bd,
                    0xf13ebd1cc7d829d8,
                    0x0a9601f7bb3b62fd,
                ])),
                Felt::new(BigInteger256([
                    0x9c6a5fc4aedc9c85,
                    0x7f2ef1336fd7d5c7,
                    0xa111cca0e75f1130,
                    0x084ae23443a12681,
                ])),
                Felt::new(BigInteger256([
                    0x8d0610673924f48a,
                    0x5515ccba80aeb3f3,
                    0x64f5073d25e05ea0,
                    0x5c9d8c5482583329,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xeab98138cdd57fe6,
                    0x935315284ef6a262,
                    0x5e3d68fb0a874708,
                    0x68842893f270490e,
                ])),
                Felt::new(BigInteger256([
                    0x7f5d9f6a3ee01da0,
                    0x84a97cf37f3e4111,
                    0xa9d63a857006a0b1,
                    0x2973f4348e28851c,
                ])),
                Felt::new(BigInteger256([
                    0xb28dadfca539ab94,
                    0xdcc70356bd651b97,
                    0x7da949a982fa4af6,
                    0x68c7ce1980403c6a,
                ])),
                Felt::new(BigInteger256([
                    0x23cc8d2742277837,
                    0x044bcbe3cc7c21ff,
                    0x33a5760e622b0914,
                    0x5095d7a544c0b7f0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x285420c23018c20a,
                    0x7898bed7a837a578,
                    0x19bd4251af054dd2,
                    0x1197463f6be166e6,
                ])),
                Felt::new(BigInteger256([
                    0xc5f13269b7e20214,
                    0xc40d7f7c5f39ac1d,
                    0x8146c047d6649cd2,
                    0x40824b48f95e5670,
                ])),
                Felt::new(BigInteger256([
                    0xed9b275193259e83,
                    0xa3d28603c97c0670,
                    0x494811edf17c793c,
                    0x066578061ca4ddd3,
                ])),
                Felt::new(BigInteger256([
                    0xe19794cc5ed41309,
                    0xbdfcc3d8f3af9c65,
                    0x47e1f86bbf6390df,
                    0x697575f24af5d268,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 2));
        }

        let input_data = [
            vec![Felt::zero(); 8],
            vec![Felt::one(); 8],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x9f3fca7d0ffe6e84,
                    0x366a4197eedfbced,
                    0x1f4e9526eabd70e8,
                    0x1bed4b74f0cb79a1,
                ])),
                Felt::new(BigInteger256([
                    0xd4ac1c6c1a8c546a,
                    0x577eb9becd963e50,
                    0xe1934fa735315590,
                    0x4a949105511214b3,
                ])),
                Felt::new(BigInteger256([
                    0xb58ad861ddc13739,
                    0xd544bc8fd08e4991,
                    0x09f8a3b0872be169,
                    0x5aaeefcc8657fe79,
                ])),
                Felt::new(BigInteger256([
                    0x5ad10af92e1bfca4,
                    0x6915295c23957e60,
                    0x58b3e30b023d8e56,
                    0x4b3c74c35fd7af57,
                ])),
                Felt::new(BigInteger256([
                    0x4c4c9e744b7549d2,
                    0x2e9f6f6a0d40b5ec,
                    0xd6b0ecb999d2259a,
                    0x54f7284ae9f917cb,
                ])),
                Felt::new(BigInteger256([
                    0xc3f99bc7b1479456,
                    0x2aea71b7e417b818,
                    0x2917e6fc86e4550b,
                    0x6e3b25b155b0c691,
                ])),
                Felt::new(BigInteger256([
                    0xc83105e0ec552134,
                    0x16495f10eca29916,
                    0x0684847305cd28d3,
                    0x67ff83c776d8aebb,
                ])),
                Felt::new(BigInteger256([
                    0x889c2c741078839a,
                    0x7567df8c240b5cce,
                    0xcc3d21841a9daad7,
                    0x7242fcd7e34170b8,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xcc0139fbe04e986c,
                    0x90865814d8c34e7c,
                    0xb4de3f223268a0f8,
                    0x5e20d6151a7b6246,
                ])),
                Felt::new(BigInteger256([
                    0xb5b881bef66ada83,
                    0x078fcfbd669c0f6f,
                    0xd872f3bc0c88a0d3,
                    0x2dd50218a6451d24,
                ])),
                Felt::new(BigInteger256([
                    0x3f0c0ca798d1a490,
                    0xeb5b08bd33b2d627,
                    0x84017a9c079d9948,
                    0x728d500d01745182,
                ])),
                Felt::new(BigInteger256([
                    0x636288c064e9b51d,
                    0x5f3f73d48dfa1609,
                    0x7004b5502d8a7d52,
                    0x185f8971946746eb,
                ])),
                Felt::new(BigInteger256([
                    0xfb3ea58fd3b614b2,
                    0xf250c4eb09c6fec6,
                    0x23eb72720e8fe977,
                    0x38bc3062a56a3e62,
                ])),
                Felt::new(BigInteger256([
                    0xf06997b1e1a338a9,
                    0x327a05ff861e7953,
                    0x63a3eab0396ed132,
                    0x0a6e9e4a12ce1cbd,
                ])),
                Felt::new(BigInteger256([
                    0xbbeaf9a82a257c89,
                    0xe1e86b4fc0b5465a,
                    0x587ce78ffb994533,
                    0x3e243fadfee86bba,
                ])),
                Felt::new(BigInteger256([
                    0x52ce75c8ce2a3cd3,
                    0x772822cb083c31c8,
                    0xca58ed2361363893,
                    0x1c3370f434cc532b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x59177821a6d1429f,
                    0xc072b1acf8b3f8cc,
                    0xe63a00a090fcc835,
                    0x433f13df27b899eb,
                ])),
                Felt::new(BigInteger256([
                    0xec1ffc66e7e88a0b,
                    0xe419620c761bb839,
                    0x3f6b8cc2f3d9292a,
                    0x294b44f1218bee06,
                ])),
                Felt::new(BigInteger256([
                    0x7eb4a06b4f7c0662,
                    0x92b71041efbde5bc,
                    0x80f2db698400a4fc,
                    0x4b17ca46ef8b77b6,
                ])),
                Felt::new(BigInteger256([
                    0xd38e202b58865acd,
                    0x28732f03f9db7671,
                    0xf0600b1d4649ef8a,
                    0x6bb2c1157646c65c,
                ])),
                Felt::new(BigInteger256([
                    0x25877ac4ffc52bc9,
                    0xf8b5062db1fa8f96,
                    0x2ed1b41ae065b7b7,
                    0x1d004f50308cebfd,
                ])),
                Felt::new(BigInteger256([
                    0x3c1c0d921903c234,
                    0xf91662da2fb3d709,
                    0xe16f8927aefd9fc1,
                    0x1539cd46d513462a,
                ])),
                Felt::new(BigInteger256([
                    0x8322e5442581b405,
                    0xeb197156cf6879ab,
                    0x5771d0b18d83242b,
                    0x39ffb0b8b2144389,
                ])),
                Felt::new(BigInteger256([
                    0x06293d150a30d7db,
                    0x0b66fe923bc62911,
                    0xf7e277b777e45887,
                    0x1d56bf6f37881976,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe24284f9afaa739e,
                    0x6d1720ee1b40368d,
                    0xa40ad9486608412a,
                    0x738c13eedd222a01,
                ])),
                Felt::new(BigInteger256([
                    0xa39b72bd131216a2,
                    0x7fb6257b0112ed73,
                    0xd446e17a5997e438,
                    0x59cd7df9113c79f9,
                ])),
                Felt::new(BigInteger256([
                    0xbfda7ad1bc475774,
                    0x7a51c8deac0b227c,
                    0x4de31997a1fa5da4,
                    0x653b8ceddcd55e00,
                ])),
                Felt::new(BigInteger256([
                    0x42204489ae2d0ddf,
                    0xfd1d65f4d533aad1,
                    0xebe8b7f459156f3b,
                    0x3b6c096272c6a761,
                ])),
                Felt::new(BigInteger256([
                    0x5007a42504cd50c9,
                    0x28123ef5403ad156,
                    0xad81d5a149a053e2,
                    0x521f69545660f1fb,
                ])),
                Felt::new(BigInteger256([
                    0x3f58c8b0914d8c3a,
                    0x2f05b873d5c633fc,
                    0x73bfb2897941b597,
                    0x4f7ac652ee95e839,
                ])),
                Felt::new(BigInteger256([
                    0x836fd84c811ae8da,
                    0xcc077f74ae595e47,
                    0xdd149e5d7fa4d54a,
                    0x44db1371171bdd6f,
                ])),
                Felt::new(BigInteger256([
                    0x389826c9c9e4b7c4,
                    0x5b54995a83a5fffc,
                    0x2b50669ea1d71988,
                    0x139c2c0e1391b26e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x1369ad10e5279308,
                    0xfb644a35c068811d,
                    0x067fedc81adad373,
                    0x4751688033d13789,
                ])),
                Felt::new(BigInteger256([
                    0x72cc37c85647a040,
                    0x10418f9df8e3f6ad,
                    0xc2cbb4ccfb1e09a0,
                    0x23ade4e188b7db8b,
                ])),
                Felt::new(BigInteger256([
                    0x7fbe6ab7737e5464,
                    0x1c6e53b77db0a92e,
                    0x20ea4ee9a97efe68,
                    0x4655bf20a6628744,
                ])),
                Felt::new(BigInteger256([
                    0xb8e2260197a1c30b,
                    0xc3d140afbbc00704,
                    0x2c453d6c2aecece2,
                    0x15cef77e9cc37362,
                ])),
                Felt::new(BigInteger256([
                    0xc6b493b4b73247f3,
                    0xc8c52b7118dfe4c6,
                    0xa8b2e4c89a3c784a,
                    0x2eaff494dfd4d358,
                ])),
                Felt::new(BigInteger256([
                    0x9ba31e7e4773da41,
                    0xbdad09aeda682b91,
                    0xb62ed3640a7ec437,
                    0x5e3b946a0eaec51a,
                ])),
                Felt::new(BigInteger256([
                    0xb27a90ff23f4d8ff,
                    0x24b2202d11e382bc,
                    0xee15a100b7bbdeeb,
                    0x6a6c9a999a9ec57d,
                ])),
                Felt::new(BigInteger256([
                    0x62ec0c8eee4c6808,
                    0x05496a89258e1f50,
                    0x3a9ecc8870a0eb1b,
                    0x2dd99ba8bdcbde77,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2d58fd876201a046,
                    0xc8257476b83f0818,
                    0x8c755ff9028d070c,
                    0x6529b653c3b7c9cd,
                ])),
                Felt::new(BigInteger256([
                    0x9ae707ca2c3e351a,
                    0xc85f37dd70381f6d,
                    0x3a3b9df02f18e90a,
                    0x473426479cd3d3df,
                ])),
                Felt::new(BigInteger256([
                    0x455e08fd8312725b,
                    0x40ca539aea757340,
                    0x2c024b2a46196c75,
                    0x0b18adbc50991946,
                ])),
                Felt::new(BigInteger256([
                    0xda4bdf62b787ab6b,
                    0x59b10fc408441d4b,
                    0x243869dd6f33c863,
                    0x07c87570c063aee7,
                ])),
                Felt::new(BigInteger256([
                    0x43f1db68de52681e,
                    0x8f4fff231f95b34e,
                    0xbed5d5f7a7c82c0c,
                    0x401aab173243d48d,
                ])),
                Felt::new(BigInteger256([
                    0x416129198a0132c0,
                    0x809a17c6595eaaf1,
                    0xce15a7ad525bcf97,
                    0x415b914572150878,
                ])),
                Felt::new(BigInteger256([
                    0xd7b6490f6efeebdb,
                    0xcc87cae7a7e41c61,
                    0x99af24142afad94e,
                    0x0791be483ceb3fe0,
                ])),
                Felt::new(BigInteger256([
                    0xcd07bb73fcb0374a,
                    0x8504dda15c40a43d,
                    0xdc9bd3eb0f09913a,
                    0x492b63a4dafcd45c,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x1f3479fd2f65eab4,
                    0xf666616bbe7551db,
                    0xa238e28142e6d6af,
                    0x262cd0a30e259c50,
                ])),
                Felt::new(BigInteger256([
                    0x05e382b1180fa369,
                    0x6aff1dada3a56489,
                    0xdf80bf1b9f39e755,
                    0x4f5a2d05253c6ab5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd4a9e22d436f92fc,
                    0x8c42c979aff58a41,
                    0x5d18496920dcc42e,
                    0x1dcbfb34d23c1d1d,
                ])),
                Felt::new(BigInteger256([
                    0x6d016fb33b152267,
                    0x4273a24131711e48,
                    0x3dd81d67d5963fdf,
                    0x5cc96a69338baf39,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4b9e5e6e6099ff3a,
                    0x69324c03e241e7a9,
                    0xba01d07443a71fcb,
                    0x51989498673efa21,
                ])),
                Felt::new(BigInteger256([
                    0x62c7a0c509d87246,
                    0x63a0c64deb9af675,
                    0xb5cdc03cb903e42b,
                    0x4a1e45511e7fd406,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x660d4ec24480a3d1,
                    0x2a9ce0602f952b8b,
                    0x5b263dc9be757c69,
                    0x097ede1a745b6d7b,
                ])),
                Felt::new(BigInteger256([
                    0x76ebbc84f1ee499d,
                    0x7c47cd7246c83b1e,
                    0xc05e0c459a57685d,
                    0x5ecebac4f2bf1478,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb4622652e0bfe479,
                    0xe7e69393c0464a43,
                    0xbb8ec7b9c90599b7,
                    0x638c714c4e82fbc1,
                ])),
                Felt::new(BigInteger256([
                    0x341c56bce9f70f5f,
                    0x3cd59afeadbebac9,
                    0x3a5f13d5079476b5,
                    0x1c3f4fb069ea9bb1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf5d83067c4da29ab,
                    0xbbcfeda209671fac,
                    0x5aa6e456ddf53344,
                    0x4e35366b8f090b22,
                ])),
                Felt::new(BigInteger256([
                    0xe5eca13467ce305d,
                    0xa635fbcdbbc72ab5,
                    0xcd2b269dcf23ab6d,
                    0x719c2f38d3f45f22,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf7fc30b32bfc12d7,
                    0xb0b6e586d575036b,
                    0x9d93ba275c043c71,
                    0x5bbc26999c852ed0,
                ])),
                Felt::new(BigInteger256([
                    0xa5adcece5539177d,
                    0x47e39ac49b4d651d,
                    0xcb7ce8961517452e,
                    0x299d073548eac313,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x964df180d218889e,
                    0xd26d923d991f7274,
                    0x0cbc5a4a060a78b3,
                    0x39badf741df7c2dd,
                ])),
                Felt::new(BigInteger256([
                    0x202d5280aee77f26,
                    0xe8692ff7c80bb847,
                    0x230241e5168fb3b8,
                    0x2bc8b5df5125bab3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa4248dda10db6703,
                    0xc962ef44fa9cee21,
                    0xb97f6971350c9edd,
                    0x5fa3acb5cbb915de,
                ])),
                Felt::new(BigInteger256([
                    0x42fa40c737cba811,
                    0x7c945580f3acfe4f,
                    0x850ba11c42776285,
                    0x4facbac630256b4c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4684cd6d5b1156a2,
                    0x8327d641cedbbc04,
                    0x138c7411f340018f,
                    0x518eda0849db9a70,
                ])),
                Felt::new(BigInteger256([
                    0x90466d3755be8b02,
                    0xd2779e5113371d8e,
                    0xb9930367a6fe8d73,
                    0x4f9b5bed514b515d,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
