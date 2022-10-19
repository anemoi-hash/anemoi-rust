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
        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
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

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn merge(digests: &[Self::Digest; 2]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        // 2*DIGEST_SIZE < RATE_SIZE so we can safely store
        // the digests into the rate registers at once
        state[0..DIGEST_SIZE].copy_from_slice(digests[0].as_elements());
        state[DIGEST_SIZE..2 * DIGEST_SIZE].copy_from_slice(digests[0].as_elements());

        // Apply internal Anemoi permutation
        apply_permutation(&mut state);

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
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
        assert!(k % 2 == 0);

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
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::super::BigInteger256;
    use super::*;
    use ark_ff::to_bytes;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![Felt::new(BigInteger256([
                0x74623e7ee84aac00,
                0xc1018f843ac1fea1,
                0x1f30c40c385f3c65,
                0x199c1a261078242f,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x810ff99343e6dd00,
                    0xa81e8ce798cbe0a2,
                    0x46cef7c10e209f8d,
                    0x1a031fd3035418b0,
                ])),
                Felt::new(BigInteger256([
                    0xfe0d5b826c34f3f9,
                    0xdc86aa28ad66426d,
                    0xf2e653250b944d86,
                    0x26455b168fd99edf,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x26b5713d041cf8f6,
                    0x0a878e957c78f66c,
                    0x24ef33167d92d2c3,
                    0x2973afcf57733f32,
                ])),
                Felt::new(BigInteger256([
                    0xac60250807d0cd21,
                    0xe217b99f175308c2,
                    0xc5da949fedb3270f,
                    0x0864a32a79a12bc8,
                ])),
                Felt::new(BigInteger256([
                    0xe818ee29a9549b1a,
                    0x0cf695c44ebcc80a,
                    0xbdaf57ca134a8a7c,
                    0x1799f7fa98009536,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0f3658181613b281,
                    0x13558c66962e0bb9,
                    0x9ff60074413a7ec0,
                    0x100eb53ef943d64f,
                ])),
                Felt::new(BigInteger256([
                    0xe85d11a8e0dad5d4,
                    0x384a047e1742d665,
                    0x403c0c27c90c29f8,
                    0x07dcb72573c797d5,
                ])),
                Felt::new(BigInteger256([
                    0x245fdbb7a3a14551,
                    0x5e86410b7b3ba081,
                    0x29f03067ee44590c,
                    0x2839b6c7e16c034a,
                ])),
                Felt::new(BigInteger256([
                    0x1888cb27213fd37f,
                    0xa54a8ad087d2eac4,
                    0xac92ece4b62caf32,
                    0x2fa0f7facb5c0503,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0c6c0e71cce9f108,
                    0x9456661ed41f023f,
                    0x1b983e02775d4ff9,
                    0x1dfdaf164efa6d35,
                ])),
                Felt::new(BigInteger256([
                    0xafc89225e90bb340,
                    0xefa8d8e258eab4c2,
                    0xe6b99a353253350e,
                    0x0bc708945904940d,
                ])),
                Felt::new(BigInteger256([
                    0xe403f68eaab1903c,
                    0x7a8f24ca855dda7b,
                    0xf1aa5a4a422b2692,
                    0x1074fc072d587e59,
                ])),
                Felt::new(BigInteger256([
                    0xadf5dfa73d40415c,
                    0x2bca118a67d1c2c5,
                    0x1270622b3e74a542,
                    0x1548579a29d0cbdb,
                ])),
                Felt::new(BigInteger256([
                    0x0cf66729c451ad6c,
                    0xcdab56fe4ffaf4e4,
                    0x351debfb018d2b8d,
                    0x1a419742ca1fe4f1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x23fb5b6d86e56380,
                    0x8cab47ddadd9b7f3,
                    0x8005babae80840d8,
                    0x0b486ab1c56c5baa,
                ])),
                Felt::new(BigInteger256([
                    0x2d3e44a47161a9b0,
                    0xc5500fda5ee28be8,
                    0x20be237d8473e086,
                    0x29e90a8b715a1dd0,
                ])),
                Felt::new(BigInteger256([
                    0x78a64436526ac10c,
                    0x5e931e89953977e5,
                    0x2b5b032c010e9bc1,
                    0x1819afb0977978c6,
                ])),
                Felt::new(BigInteger256([
                    0x541ca92eb0a7a33e,
                    0xd124b28fef7e2bd9,
                    0xc2621dae24416fd1,
                    0x11dc0133edb60ed0,
                ])),
                Felt::new(BigInteger256([
                    0x0f5abf33829894ed,
                    0xd9a4422236863f4c,
                    0xbfd1afdfcf857735,
                    0x2a323b0b3e7f10d1,
                ])),
                Felt::new(BigInteger256([
                    0x5353269b816fc5c4,
                    0x004abcf4f5ad54f8,
                    0xb8eb1e34678e6361,
                    0x0109c4fd538064b0,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x5051aefe0c82744e,
                0xec2c79cafca347fa,
                0x9a2ba71bcc6459ab,
                0x0cb9deb6d138ac26,
            ]))],
            [Felt::new(BigInteger256([
                0x09ac36a6c6da39d4,
                0x332eac852d6ab048,
                0x571d7e9168693f93,
                0x045745b48c32160c,
            ]))],
            [Felt::new(BigInteger256([
                0xf26591d4922b9314,
                0x9c9d7b3560247280,
                0x2f4fa2afb09499ec,
                0x20fb339d89dfb6b6,
            ]))],
            [Felt::new(BigInteger256([
                0xf260335cb00e36ad,
                0x9fb487578a867da5,
                0xcaa8fe7f019540b0,
                0x1e2f6d12b9174a67,
            ]))],
            [Felt::new(BigInteger256([
                0x448ecb71b9c1b14b,
                0x79638fc2a07687ac,
                0x465e80f9c5f9972c,
                0x04ca025e9e598131,
            ]))],
            [Felt::new(BigInteger256([
                0x4cb1304ebf89f4f1,
                0x3ad74369adb144fd,
                0x11b4d9555eecb425,
                0x0b6f728f9e093f50,
            ]))],
            [Felt::new(BigInteger256([
                0xff165d4cd55c4b89,
                0x9757604bd22cfe67,
                0xd8783c3a551d87b7,
                0x10cfd79d2cfdbd1b,
            ]))],
            [Felt::new(BigInteger256([
                0x2cf41f2326b91aa2,
                0x11576e25b781b8a7,
                0x925ddeb513aade27,
                0x1c75baa68c409df3,
            ]))],
            [Felt::new(BigInteger256([
                0x020b08486268a23e,
                0x0cdde53263031bad,
                0xeeb67e4d9a7c569f,
                0x05303efd5a8f76f9,
            ]))],
            [Felt::new(BigInteger256([
                0x23733295eb84b5eb,
                0x27a14bac44118c5a,
                0x0fb094926e2d05b5,
                0x0ba3ff488eedbe72,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_hash_bytes() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x5051aefe0c82744e,
                0xec2c79cafca347fa,
                0x9a2ba71bcc6459ab,
                0x0cb9deb6d138ac26,
            ]))],
            [Felt::new(BigInteger256([
                0x09ac36a6c6da39d4,
                0x332eac852d6ab048,
                0x571d7e9168693f93,
                0x045745b48c32160c,
            ]))],
            [Felt::new(BigInteger256([
                0xf26591d4922b9314,
                0x9c9d7b3560247280,
                0x2f4fa2afb09499ec,
                0x20fb339d89dfb6b6,
            ]))],
            [Felt::new(BigInteger256([
                0xf260335cb00e36ad,
                0x9fb487578a867da5,
                0xcaa8fe7f019540b0,
                0x1e2f6d12b9174a67,
            ]))],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 124];
            bytes[0..31].copy_from_slice(&to_bytes!(input[0]).unwrap()[0..31]);
            bytes[31..62].copy_from_slice(&to_bytes!(input[1]).unwrap()[0..31]);
            bytes[62..93].copy_from_slice(&to_bytes!(input[2]).unwrap()[0..31]);
            bytes[93..124].copy_from_slice(&to_bytes!(input[3]).unwrap()[0..31]);

            assert_eq!(expected, AnemoiHash::hash(&bytes).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![
                Felt::new(BigInteger256([
                    0x3ba82a5ae18da706,
                    0x3d6dcbe91f34b94c,
                    0xb1252903667f650d,
                    0x2fa8d0a82a8ca163,
                ])),
                Felt::new(BigInteger256([
                    0x223485058bf2d40d,
                    0x79e0399dbbab9ce6,
                    0x4ccb36d92fa96c17,
                    0x03c4a060490b4939,
                ])),
                Felt::new(BigInteger256([
                    0xd7a784a4bd2c166b,
                    0xd28cf7edb3b8915c,
                    0xe7b2037ff56898a6,
                    0x1e8d3d01c862caba,
                ])),
                Felt::new(BigInteger256([
                    0xe80790e081f131fa,
                    0xe6a032fb8385710f,
                    0x2bd00eb1ed0f451d,
                    0x15dc919cabb5ec0e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x9c5752618791ece9,
                    0x5d03d8ff07c2931f,
                    0x3677b567006d0a61,
                    0x253d26ad17472071,
                ])),
                Felt::new(BigInteger256([
                    0xf976b04e2c3ff157,
                    0xb9016e54d91953c4,
                    0xf25f1eb01a9e1026,
                    0x016172ec02cefc46,
                ])),
                Felt::new(BigInteger256([
                    0xe890298da616dd52,
                    0x9df50b7b0a62f5ba,
                    0x0b31d5712a1cd240,
                    0x0eb2fbf07d8d6039,
                ])),
                Felt::new(BigInteger256([
                    0x56a134528a5968a1,
                    0xa354e85e2dfd3e24,
                    0x6d823725ee5bb8f7,
                    0x1984ff128960394e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x40c468ddab6e2412,
                    0x0207beb9ee2e2cee,
                    0x0df32fef9d65b798,
                    0x2392d8c075f74ffb,
                ])),
                Felt::new(BigInteger256([
                    0x4902d92fd236056f,
                    0x6dc20e8e70ecfc58,
                    0x99063134ff656525,
                    0x140d89986895b7c5,
                ])),
                Felt::new(BigInteger256([
                    0xb03cede521e7d373,
                    0x837fcc5feeb60249,
                    0xbed41c615559205e,
                    0x28e411c11849f568,
                ])),
                Felt::new(BigInteger256([
                    0x65e656cb6fb9b2ed,
                    0xd3f90cfab1de95d6,
                    0x16b66fee939a19c4,
                    0x025e02c22cf8dc83,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x214783a83af9d3bc,
                    0x9db744d7cd7670ab,
                    0x61a6d6c05b8c790b,
                    0x0c3725f469b88f62,
                ])),
                Felt::new(BigInteger256([
                    0xb797ecb818c4bbae,
                    0xf61f915c43e8e0d1,
                    0x5da4a5a9b41ec2af,
                    0x15d0b85ca689fabc,
                ])),
                Felt::new(BigInteger256([
                    0x6dbdb46245f7ecae,
                    0xd262ee6902853e8c,
                    0xfaf7db95a60bb8ef,
                    0x081242e1de1401f7,
                ])),
                Felt::new(BigInteger256([
                    0x8aa87b750a9b03bc,
                    0xf2dfea937f63374f,
                    0x6fc32db882ea6e01,
                    0x10fa2aa99e694eef,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3562fbf67c6e0188,
                    0x858f5c41105d3651,
                    0xf90d9be9fca96dcb,
                    0x0442bbefe6f38923,
                ])),
                Felt::new(BigInteger256([
                    0xf730d2dbfc8ba52d,
                    0x4132c40fa830bbd5,
                    0x8a12514baf8d795c,
                    0x03a820c3e02d7c4f,
                ])),
                Felt::new(BigInteger256([
                    0x499da62dc7428569,
                    0xeb738c6c66944202,
                    0x427597e905e32d31,
                    0x1534ddd1b09e823b,
                ])),
                Felt::new(BigInteger256([
                    0xd99816af8620fc08,
                    0x1e5a483e6575ee9d,
                    0x6d90abf46e859f64,
                    0x11aab9ed8fb9f0a3,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x1b075e5ce0401d28,
                    0x29fce63863835f6a,
                    0xcb16cf5d6475e47e,
                    0x2d39cd90534dd44d,
                ])),
                Felt::new(BigInteger256([
                    0x1737b244bd925432,
                    0x6cb66844715ed08c,
                    0x45a95e530736bd87,
                    0x00f7ab78ab35e38c,
                ])),
                Felt::new(BigInteger256([
                    0x10b6d373c1db423c,
                    0x4056e664fa65c373,
                    0x8973c3801be03f7f,
                    0x12469df12540cb8a,
                ])),
                Felt::new(BigInteger256([
                    0xdae7f84e1beb65ad,
                    0xedd2926ca0a0ebe8,
                    0x20ff18c4c560d0c2,
                    0x263f1c1638e5a850,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x52f3b9073aa01b64,
                    0x46eb7a3aae38cc7a,
                    0xb885bc052dbb1267,
                    0x2609778f12b8b062,
                ])),
                Felt::new(BigInteger256([
                    0x368f2f734fd95a4d,
                    0xa358c684e02ba1ee,
                    0xbae2089f48c54544,
                    0x2a4ca9ccf61aae02,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x45a517360dd8e5ec,
                    0xee4039e91cdacaa8,
                    0x1af03253c8f46181,
                    0x05aff23550b3ec8e,
                ])),
                Felt::new(BigInteger256([
                    0x10ba57993382f40c,
                    0x1c70c04eb22a6872,
                    0xa673f80be5dbbeaf,
                    0x2246fed0d555cd19,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x67fd607b6a9d481b,
                    0x4b60ff9689c67fd7,
                    0x0da2a938c03a1712,
                    0x137edf3e4ef28e3b,
                ])),
                Felt::new(BigInteger256([
                    0xe196189b13f0144e,
                    0x066f88e4581b48bf,
                    0xf594bc6f0a240fe1,
                    0x1c65d2f993ff55f1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4d0d02b9b5a56d18,
                    0x7b44b60ba34d7853,
                    0x768385795451f581,
                    0x1b004de414bf0043,
                ])),
                Felt::new(BigInteger256([
                    0x315634d1f62aaf0c,
                    0x07d314c120c585fd,
                    0x6a32432b1c0c8a41,
                    0x13b73b14bd3b1c59,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd6ab1225957800ce,
                    0x1078759e96a7c20d,
                    0x62b88076c1bcedcd,
                    0x2854b0ab37848115,
                ])),
                Felt::new(BigInteger256([
                    0x3bba48d1b207b679,
                    0x94d2068a569514c9,
                    0x30ba5f749065899b,
                    0x0c62336a4ddb85d7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x19d6eeb5f8c16001,
                    0x1eefe43cafecd210,
                    0x5ab881a582021622,
                    0x2f12232a15387ea0,
                ])),
                Felt::new(BigInteger256([
                    0xbf3f62063378e821,
                    0x76ca04cb9e1dd010,
                    0x131b715630140f40,
                    0x102cf8d7009b3c1a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xfd44532a186deb25,
                    0x496ebd7806c5e2b3,
                    0xca5d7481ee7e1592,
                    0x047cc6f9bf38f8b8,
                ])),
                Felt::new(BigInteger256([
                    0xc2fe52418cf3878c,
                    0xde39d9c5b8f2739b,
                    0x51a7966b2aa4a22a,
                    0x2dfe5e16edb8e9e2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9d9dad1f27234168,
                    0x05f4eea40e129e86,
                    0xca34ada1f258a1da,
                    0x1d472dda90d07b0c,
                ])),
                Felt::new(BigInteger256([
                    0xb318ab0c8ffde91e,
                    0xcc0e5559e9f48842,
                    0x60a210f51f154d4f,
                    0x2a077514c36db433,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe1d84cb006c13cee,
                    0x4ae90c4c46597c91,
                    0xdbe09f1e4a659081,
                    0x25e814377e4f5b72,
                ])),
                Felt::new(BigInteger256([
                    0x77f2aabd7b35a1a1,
                    0x1b66d057f377405a,
                    0x6ea9e237c139b74d,
                    0x03da9706eb25fbe3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8d699667e148c783,
                    0x37a81c072890be80,
                    0xe821aca530b0e6a8,
                    0x06a85fa1622a8957,
                ])),
                Felt::new(BigInteger256([
                    0xb9460b0f72c42d0d,
                    0x0350379a88991f77,
                    0x37cb2ce49359eca5,
                    0x08c7815c2a01fcdb,
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
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![
                Felt::new(BigInteger256([
                    0x5258bb4af0e681c3,
                    0xcfc23e37bcbc198f,
                    0x585236e010e21bb3,
                    0x01a419e62b1594f8,
                ])),
                Felt::new(BigInteger256([
                    0x9a10420c85bc68be,
                    0xde17d30a3c355368,
                    0x77b6752160eaa3dd,
                    0x19fc129eee562178,
                ])),
                Felt::new(BigInteger256([
                    0xc23796a6e929e214,
                    0xa902cef8260534a2,
                    0x1031a31025f2a740,
                    0x15f06002faa33b5e,
                ])),
                Felt::new(BigInteger256([
                    0xaca801fd2227eae8,
                    0x737518a19e366392,
                    0xd7cf606b04708eda,
                    0x09ce8d6a5ce123d1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x85335ac035c3b857,
                    0x15589039ce6f4371,
                    0x1f89947efe6800a2,
                    0x15c1042e23a2ba57,
                ])),
                Felt::new(BigInteger256([
                    0xcbd987ed3c15184f,
                    0x58c5cc8e3c687261,
                    0x591387d591bc2d1c,
                    0x05cfb6582bdcdaac,
                ])),
                Felt::new(BigInteger256([
                    0xc6f7ac1accb5a7eb,
                    0xcad46c97e59bbb7c,
                    0x6c6dc2dd718bde76,
                    0x275c6ef7b7220f15,
                ])),
                Felt::new(BigInteger256([
                    0x9918fe144c2a4a47,
                    0x6b26b7aa5df5afe8,
                    0x2762c51b7548f45a,
                    0x125f6a9bc7b9a657,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x1ae90b70d840b45e,
                    0xe80c03a0200e3ba4,
                    0xf5ee539b0188d1ec,
                    0x069d71dbeeb93ba0,
                ])),
                Felt::new(BigInteger256([
                    0x1845c26c61a802f4,
                    0x08f4c81d29cdf82e,
                    0x6b0cbadfdc8efbf4,
                    0x22f851b8edb6f1bc,
                ])),
                Felt::new(BigInteger256([
                    0xbb102b2e8d1b929e,
                    0x31dadd0c1c37214f,
                    0xba57b67e7dd7766b,
                    0x14611b67ce2f490e,
                ])),
                Felt::new(BigInteger256([
                    0xc72ae1835f030a5d,
                    0x9fafa93eae4eb3f9,
                    0xaa6bbce6639e353e,
                    0x096496538e43d537,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x20f1fa5619560be5,
                    0x0a6c74852bb7ad25,
                    0x5243f1b5386601a2,
                    0x11105881387ad4dd,
                ])),
                Felt::new(BigInteger256([
                    0xf30b59d0728d26e9,
                    0x334c99a67194de56,
                    0xf6379a441aef1953,
                    0x03259ce430d192d0,
                ])),
                Felt::new(BigInteger256([
                    0x8c3719dc9a4e5221,
                    0x4d9e476f5ecb5e88,
                    0x12ff01f9fd9d45d4,
                    0x1e052264f2bc2519,
                ])),
                Felt::new(BigInteger256([
                    0x99fca54e2b1d6127,
                    0x52aae24ca7e80d11,
                    0xe08d32f635b73749,
                    0x0999b97ecf526aa2,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xaabd04b306eb2cf6,
                    0xc694fef816ffc0fd,
                    0x2dfe9a3d77e38378,
                    0x2650565342c9ca23,
                ])),
                Felt::new(BigInteger256([
                    0x800b3ede2bd5af78,
                    0x7771b696a151dd7f,
                    0x07d8cc7fff501970,
                    0x1b2c4468c0d05dbf,
                ])),
                Felt::new(BigInteger256([
                    0x74849cda8e7ad635,
                    0xeb95fc72e0725bac,
                    0x5ce16a394fec91d0,
                    0x2f3149e748a54125,
                ])),
                Felt::new(BigInteger256([
                    0x9e312a4d8589c1bf,
                    0x3e7f32f1ed5f277c,
                    0x603d8b88f03fed98,
                    0x1d805e36e6e224f4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc3f51c70b81eb269,
                    0x66f085e2826e828e,
                    0xe426af50b859b422,
                    0x18c3db4efc0826c7,
                ])),
                Felt::new(BigInteger256([
                    0x4bfdaa777a0a60dd,
                    0xf733c8cffeb8b694,
                    0x1fc5f1cd796d7f04,
                    0x14d507255c9025cc,
                ])),
                Felt::new(BigInteger256([
                    0x605ba7b748346fa7,
                    0x8bd5a206eb01ca2a,
                    0x7e60bd3b8a01bc07,
                    0x175d1ad02535f999,
                ])),
                Felt::new(BigInteger256([
                    0x46af16e8c79a084b,
                    0x4d62797f8f932396,
                    0x0e59e73e1e49b5c0,
                    0x07c53f92dd88f559,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x4d625c63b1fc786a,
                0x52c2d62e25f2a3db,
                0xbb177eedf4feff4e,
                0x1ff1d2e927a1be3b,
            ]))],
            [Felt::new(BigInteger256([
                0x565f6ecf415bd9f8,
                0x0ab0fa37cf05331a,
                0xc1642a5faed02031,
                0x27f6f1062609b9a7,
            ]))],
            [Felt::new(BigInteger256([
                0x499379167e8d5c69,
                0x51d0887ae1e1c897,
                0x033765a7ca5e26f3,
                0x2fe4b237e2f1e42d,
            ]))],
            [Felt::new(BigInteger256([
                0x7e63378babd01c24,
                0x8317caccc412fe50,
                0xe0b5c8a4705e7fc2,
                0x2eb788f8d1fa1c9c,
            ]))],
            [Felt::new(BigInteger256([
                0xf2514c4c6cd0467b,
                0x4955eb30bbc0eebe,
                0xe03c58ec0f8cdaae,
                0x0f3e79e97382310a,
            ]))],
            [Felt::new(BigInteger256([
                0x4594a836cc3cb33d,
                0x59739791102020b3,
                0x87b63cb87a922ff3,
                0x106cf225d6e64697,
            ]))],
            [Felt::new(BigInteger256([
                0xf2d0dba10e0c32bb,
                0xcac1649df7cb79ad,
                0xa83c90d819beaa98,
                0x2fde141a50c5d283,
            ]))],
            [Felt::new(BigInteger256([
                0xa49c12aa127267fd,
                0x72e91d8d8a7677de,
                0x641981cbdc8bfba9,
                0x1ad75ba00e4564fb,
            ]))],
            [Felt::new(BigInteger256([
                0x113d0ba5d50646e7,
                0x90ae59ca3b8bf060,
                0x4c812d785eca549b,
                0x0455462b418c1182,
            ]))],
            [Felt::new(BigInteger256([
                0x16384cc3531f3134,
                0x10479fa7a0b6f1ac,
                0x91df1d20705571ab,
                0x2d5461e2db2b333a,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
