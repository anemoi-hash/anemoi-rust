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

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![Felt::new(BigInteger256([
                0xcfaa16031d08a458,
                0x2327ee7ef6e70a29,
                0x1e96214565bed8cf,
                0x33431cef1c32a1be,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x6fef07485f5a1c4b,
                    0x04ac863e1cb6c1b0,
                    0x3dd3bba34a965330,
                    0x017384b890030a09,
                ])),
                Felt::new(BigInteger256([
                    0x8466efd9b2fbbd32,
                    0xb66bf68b44c12768,
                    0x727663dfb7f7e667,
                    0x3c2664e482b86dd3,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x24c2fce7c6a8f58d,
                    0x70ee26a7e5b7f758,
                    0x440c720d1f038209,
                    0x1cbaabd7015dfa70,
                ])),
                Felt::new(BigInteger256([
                    0x89ddee939a42b7d8,
                    0xb6c39dab6751569b,
                    0x976f185988b95473,
                    0x0be5e32e52d11db5,
                ])),
                Felt::new(BigInteger256([
                    0x5115724c68343f40,
                    0xafd2841b1c81eecb,
                    0x48377ad810cd15f6,
                    0x300e00f74b317eb0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe25011575789c6c4,
                    0xec33d619a527d545,
                    0x4f295c7321e3b6ae,
                    0x261d4b3058082036,
                ])),
                Felt::new(BigInteger256([
                    0xcbf3a6e058541430,
                    0x84c49ccdba4cf888,
                    0x9f3261600dea4fe9,
                    0x0e0fbeb5baf3ac6c,
                ])),
                Felt::new(BigInteger256([
                    0x279aac18aab92226,
                    0xa4cac0bd8f420b67,
                    0x19ffd75d2db8f37a,
                    0x0eaf2f0b1c621b04,
                ])),
                Felt::new(BigInteger256([
                    0xe59e8ba42e25825e,
                    0x748381eacc38ee0c,
                    0xcfd935cfbca98387,
                    0x29747fa8a6f2c6b0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd839e16303e373e6,
                    0x27b3d198a0be98e4,
                    0x2940ba68f26d2d1e,
                    0x1d8a403982f8a3b4,
                ])),
                Felt::new(BigInteger256([
                    0x250c102406b7ca8f,
                    0xec33df1cb8f43a34,
                    0x3a91819ab41acb4a,
                    0x2c417d741c5adcc0,
                ])),
                Felt::new(BigInteger256([
                    0x1835d3bd08661c12,
                    0x2118e940ed1add23,
                    0xaffdd026bd0d4a0c,
                    0x06709dba3409926e,
                ])),
                Felt::new(BigInteger256([
                    0x2bb221e9df8af5f8,
                    0x3ac6256930ac95ed,
                    0xc3a0f8787aab5f55,
                    0x0833b7c1c014ac3b,
                ])),
                Felt::new(BigInteger256([
                    0xb59eb8b0dc1e1d19,
                    0xc62a37b3af44393d,
                    0xa1f99f9da2c80930,
                    0x3e399e98a32ce9df,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x70c988eb7fb824d8,
                    0xf25d41e16f356de2,
                    0x0b5538a23f9f4ac0,
                    0x2c8e4e34e0113af3,
                ])),
                Felt::new(BigInteger256([
                    0xfde988fa96d49f03,
                    0xb4d88e542cddc8ec,
                    0x8da509dadc26c80b,
                    0x3e5d7fa9bf41c3e4,
                ])),
                Felt::new(BigInteger256([
                    0xb6598345e98eec85,
                    0x746998298ec71525,
                    0x9ffda5a3d8e5d55d,
                    0x23fc37f84e323e89,
                ])),
                Felt::new(BigInteger256([
                    0x49c19cd306a3fd99,
                    0x59794fd7c227a34d,
                    0x1ae88efbffcecf17,
                    0x0babfaa1d4a277cc,
                ])),
                Felt::new(BigInteger256([
                    0x4e717482e0f41224,
                    0x5e3f2599b6998db7,
                    0x19729b8073fa7602,
                    0x323606c539c6ef5f,
                ])),
                Felt::new(BigInteger256([
                    0x83f48babf95190ca,
                    0x16dded060478b3db,
                    0x74c4a3c62f4bd04b,
                    0x08ee090541ad3cd9,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x090e409aaa4512ee,
                0x6fc3f2d00aa18a39,
                0xd67c2554b8cf3365,
                0x01148784f2eba697,
            ]))],
            [Felt::new(BigInteger256([
                0xf7dacc8f184b66be,
                0xfd5700fe25ba7975,
                0x3851da11a5127658,
                0x1c52f109bfcb1372,
            ]))],
            [Felt::new(BigInteger256([
                0xf3e695d4790ade89,
                0xc211e33d4ce1c325,
                0x11f1dac08510a4eb,
                0x1a2e0cc9ff2c6778,
            ]))],
            [Felt::new(BigInteger256([
                0x211424643a6e61cd,
                0xb30cbb2669818e4f,
                0x54ae9dc8b2d1a007,
                0x25e2905173f66696,
            ]))],
            [Felt::new(BigInteger256([
                0x54a1071775a5e984,
                0xfc9bde2dc020c13a,
                0x06788c66dd582be0,
                0x3076319b8921adb3,
            ]))],
            [Felt::new(BigInteger256([
                0xd7568050bafbead0,
                0xc223607f53ac7796,
                0xa5e817c36eab6cf2,
                0x0655025d22419a8f,
            ]))],
            [Felt::new(BigInteger256([
                0xee00d1e19090fecd,
                0x8ef489ec024902a1,
                0x1e97351148995bdc,
                0x07b05d2e7f3cd438,
            ]))],
            [Felt::new(BigInteger256([
                0x0cf69aef2be6a9bb,
                0xa60069595e9765c3,
                0x3905bbe21b2be10d,
                0x0cf411dbbe826b78,
            ]))],
            [Felt::new(BigInteger256([
                0x97e36c63ca98b5ca,
                0xdcbd4c076b24f58a,
                0xbb33fe789ae91c6e,
                0x378b1ade4d63d9c0,
            ]))],
            [Felt::new(BigInteger256([
                0x9c773571b6da3204,
                0xcc2615cb0310d65e,
                0x83b767d03516e7b5,
                0x117432d65b7444ef,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
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
                    0x98fe8f81be495f19,
                    0xa535bef7fbb16f3c,
                    0x88e0deeabd77c532,
                    0x2e3997b224a568ca,
                ])),
                Felt::new(BigInteger256([
                    0xc41f46a551786b8b,
                    0x74d80ada8f7888fb,
                    0x746f789191902b99,
                    0x33ce9c525a5cae57,
                ])),
                Felt::new(BigInteger256([
                    0x90fec1158b25a948,
                    0xa494fe6c386647cd,
                    0xc6e596199271e827,
                    0x314d5fc6f2198e2a,
                ])),
                Felt::new(BigInteger256([
                    0xa4aea34b68c6a4d6,
                    0x7335ab133210c5bc,
                    0x8fe47f99a724e6be,
                    0x209abe9a17f60fec,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x10e7632a34b7f657,
                    0xf3307e5a0316fae8,
                    0xddfab4ed115fab59,
                    0x18ad59dbad9866f6,
                ])),
                Felt::new(BigInteger256([
                    0xc7840abfde30e6a6,
                    0x41f4cb23bd6b9a86,
                    0x5941c4172b28b669,
                    0x3980af3a81443615,
                ])),
                Felt::new(BigInteger256([
                    0x9a5be0e8cc69d2c3,
                    0x58afcd24703d03c8,
                    0x2c00a0b33e546601,
                    0x129f4782d4e50d30,
                ])),
                Felt::new(BigInteger256([
                    0x4d7dec0091ca3e15,
                    0xcbac5d46e8fe6248,
                    0x602aee9dd47b2f46,
                    0x1205f49978dae00c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x694a3673d121460c,
                    0xda3c485914edf263,
                    0xb0d62b3650a04fc3,
                    0x172ce595825c4964,
                ])),
                Felt::new(BigInteger256([
                    0xb735d79f522e87ba,
                    0xa6b324310d1d28cb,
                    0x8274bf8fe5b43d15,
                    0x3e9c4865a488d3cb,
                ])),
                Felt::new(BigInteger256([
                    0x7b57c110006e40f4,
                    0xf9470ea8a1b222a9,
                    0x7193da588b49eb93,
                    0x0f1307bec3c977d3,
                ])),
                Felt::new(BigInteger256([
                    0xf67c87f79b274b19,
                    0xe3a27535178493a3,
                    0xec3898efdf2a606d,
                    0x099008103a36199c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc4cb906e8b6c09be,
                    0xdc13a972e2bed853,
                    0xa51c816ac76d42e6,
                    0x301870bcd6ecefad,
                ])),
                Felt::new(BigInteger256([
                    0x1d158e29e26b11f8,
                    0xb9d95938360920aa,
                    0xb6e5b744224c0f90,
                    0x0bc1ac2863255937,
                ])),
                Felt::new(BigInteger256([
                    0x4aeec248a3691e1c,
                    0xfe31b5d28098c0a1,
                    0x7e1dba63db9cb54a,
                    0x259861bee20b5d60,
                ])),
                Felt::new(BigInteger256([
                    0x5448489ad6d507f9,
                    0xa57eb498a04c2477,
                    0xcf925b7f01b98b6d,
                    0x2eaaad9781072f5f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xfde20d38c8a47700,
                    0xd1a0bf59f91dc376,
                    0x2ae6b0dd8f2ef88b,
                    0x319e664edc60fcc9,
                ])),
                Felt::new(BigInteger256([
                    0x977f86cbdccbdf71,
                    0x9624d50e448dbb9c,
                    0xfe9ed71c44237fc8,
                    0x2bebf5892656ecea,
                ])),
                Felt::new(BigInteger256([
                    0xfd3c19c71d2e6bad,
                    0x7ae72d65e2b08647,
                    0xa4ffd26560c6db64,
                    0x09f8fa05b5dd1e1a,
                ])),
                Felt::new(BigInteger256([
                    0xf3e1342d8fdc7a3f,
                    0xd6c3d4645e7e80ae,
                    0x26293d6210fb04a8,
                    0x30414e8b8de0e5a2,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x4fd5c8a4a95c2fd9,
                    0xe6d37a9ac29aa438,
                    0x52484887f1a8b2e6,
                    0x2affff6adc30fad8,
                ])),
                Felt::new(BigInteger256([
                    0x07dc27e825be36f2,
                    0xf343dd719952c96c,
                    0xcc1c58b60c34c889,
                    0x02613f10c662bd1e,
                ])),
                Felt::new(BigInteger256([
                    0xb0eb76858657bbf0,
                    0x3e5f20dd43e18547,
                    0x313df4e4a360ff28,
                    0x0b8e1626c166549a,
                ])),
                Felt::new(BigInteger256([
                    0x131362533495d4a7,
                    0x0a5b620e39ca20c5,
                    0x51def252bb9f2207,
                    0x30cb9830320c194b,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xe14eae188a2d7dfc,
                    0x56f6540d4eda001d,
                    0x346bb7b62ae17b64,
                    0x28e6beaa4dad8558,
                ])),
                Felt::new(BigInteger256([
                    0x50f4989de9587a11,
                    0x09a5fd5b224f3a8d,
                    0x7faa9439a73b2aaa,
                    0x250d15a6e4eb5a4d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4f03191cedd83453,
                    0xc5012656946ce3ba,
                    0x4a24b33497705536,
                    0x0715f2f819bc260e,
                ])),
                Felt::new(BigInteger256([
                    0xa3dd82681ce365e5,
                    0x68dfa03db6b7690e,
                    0xf420d974943597a5,
                    0x2cc231c379fafeef,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbe2fdfbdb6ddec6d,
                    0xb2c7619e72a1573e,
                    0xe8cd97c6e0c7ee9e,
                    0x33df7b2ae526e67d,
                ])),
                Felt::new(BigInteger256([
                    0x76bcad9396d49901,
                    0x8ed1ea65ef387636,
                    0x913408d5cb25de69,
                    0x3a1bd27bc0498365,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf72a69397bdb2c1c,
                    0x3aa59a0871c51209,
                    0xed3e5b449955abd9,
                    0x23093b3e7a521354,
                ])),
                Felt::new(BigInteger256([
                    0x6279ae1edc291103,
                    0x9b9a8454ae5ab985,
                    0x13c7c7bd5d1550ef,
                    0x04369bff99b5869f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x10245abdd62b5a56,
                    0x90de49066e0c6c27,
                    0x66ec3301ea3d950a,
                    0x38ee0c96bc7cd5c4,
                ])),
                Felt::new(BigInteger256([
                    0x8c734d3dd604264d,
                    0x6497a0c28b91f522,
                    0x270a566f77cab64c,
                    0x30be139ec6c9f942,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6a26f0bc58d9a332,
                    0xfc478b0c149b2e80,
                    0xa274f80e0ec1425d,
                    0x1bce72c383de38ba,
                ])),
                Felt::new(BigInteger256([
                    0x7c6f56f5a9b36406,
                    0xa0073c88aa2f6c49,
                    0xd1a4c39e4141e91d,
                    0x2d906df236c7dcbb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x025db6aaa0cac003,
                    0x9ef490e19cc6645f,
                    0x2d4fee8e1009d314,
                    0x1e1278d7471f0111,
                ])),
                Felt::new(BigInteger256([
                    0xd44292b7967356ae,
                    0x6297f0da57402071,
                    0x89b307bbce716dbc,
                    0x1dbebcc1023e094d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd7cee912c2086531,
                    0x8b61868783fed9df,
                    0x49487da56346140c,
                    0x3426d6924d52e940,
                ])),
                Felt::new(BigInteger256([
                    0xaeca6ccaf69c88fd,
                    0x1f994b9c2232b776,
                    0x98444aa44b0bece4,
                    0x3eb01c0e6ddcbbaf,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x136dc09f2159b0d3,
                    0xcea8cb850f57877f,
                    0x3379471eeadfe73f,
                    0x0ff0980623d6af5d,
                ])),
                Felt::new(BigInteger256([
                    0x99c978114b9271b0,
                    0x05025258c5fbd088,
                    0xa59b4a4d095584c1,
                    0x0fc65116e03af930,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5b2fb142a8c14d2c,
                    0x1d324c3e3f84393e,
                    0xc6c1de60a5ada39e,
                    0x3ef735abf79dd2d7,
                ])),
                Felt::new(BigInteger256([
                    0x0d366e4393caa7b3,
                    0x11489cdb85c2b377,
                    0x5b19d6e7e75b0d74,
                    0x16f2bb9b07ccf25e,
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
                    0xd21a6d58de2fed9c,
                    0x1f582d867634f9e0,
                    0x718611bfed5466b1,
                    0x1421b2f142795265,
                ])),
                Felt::new(BigInteger256([
                    0xd9fe04abcec3e489,
                    0x680c9951d1defc24,
                    0x40e71dc395acf0a6,
                    0x2f6d75622d9c5950,
                ])),
                Felt::new(BigInteger256([
                    0xaff8117bc678fce4,
                    0xf4530cd395a7d86a,
                    0x28dd4bd2d458e91c,
                    0x22a62bc6d7f2ed84,
                ])),
                Felt::new(BigInteger256([
                    0xd4754865fa064a9d,
                    0x7677098a1812a10e,
                    0x0f114347badcabdb,
                    0x1332c692fa76758e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xaafaf1599e8fc431,
                    0x4913ee96f5c48e8e,
                    0xbbabb8b3c3d8973b,
                    0x15ebd938d41f2f7c,
                ])),
                Felt::new(BigInteger256([
                    0x31296a6a7d16fb44,
                    0x56e17ecad3d7e532,
                    0xe73568a1d628c34c,
                    0x2ae307ff29822c3f,
                ])),
                Felt::new(BigInteger256([
                    0x0d8127133ff61398,
                    0x32ce77bb71eb0c63,
                    0x8fef6dc3e2c5ef40,
                    0x2f96fc16a8aa1fc9,
                ])),
                Felt::new(BigInteger256([
                    0x7bdf439c5dfa35e3,
                    0x076e24935161d238,
                    0x19e6b5428bfc81d0,
                    0x15b3e793e456b25e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf7e0e1b1e6a47347,
                    0x774bf419a2ab03aa,
                    0x3e3671ca32c96314,
                    0x29364cbfe5d90387,
                ])),
                Felt::new(BigInteger256([
                    0x62b82e9c760faef6,
                    0x8d261663fde49b67,
                    0xcbb70c4fe82cc0c7,
                    0x0e0cf783aeb1c28f,
                ])),
                Felt::new(BigInteger256([
                    0x6ab0259829fb2a76,
                    0x426ad94b43fef30a,
                    0x13174b1d25f6bf56,
                    0x1c1b3ac5301b0124,
                ])),
                Felt::new(BigInteger256([
                    0xfed1da74be9b6712,
                    0x01e4c1ac2a2ede84,
                    0xa15eb510f5bd6a56,
                    0x34d4bd628d6fe144,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3ae38b64f8dc5350,
                    0x146729abc0bd927d,
                    0x6b85777ee8227424,
                    0x0321986c18c5bed7,
                ])),
                Felt::new(BigInteger256([
                    0x17005543f821cb46,
                    0xf0a7435948d2c503,
                    0xe989f1d7af44dd2e,
                    0x3b2c43256b2649e9,
                ])),
                Felt::new(BigInteger256([
                    0x6a3dfe9536c6b21f,
                    0xaf51d87c234765a1,
                    0x75ff263409b57b8a,
                    0x0cf4fa8fbec0016a,
                ])),
                Felt::new(BigInteger256([
                    0xae49e73d75bbb856,
                    0xccd1d073e8be7065,
                    0x31f8da9545499528,
                    0x1f70b32bbd501788,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x547caf3e105acf14,
                    0x831d6fe02d969ed5,
                    0xf0fbba5084c0e4a4,
                    0x02844987050c9e3a,
                ])),
                Felt::new(BigInteger256([
                    0x1a9a95db9302ffa8,
                    0x20249ffeeceeeb33,
                    0xb1505f556ffd4a80,
                    0x03b8408d436d7b0e,
                ])),
                Felt::new(BigInteger256([
                    0xed6047eb2efa1c65,
                    0x0537737d5210a201,
                    0xf540ccc3ba10ca61,
                    0x129303abfcc3b865,
                ])),
                Felt::new(BigInteger256([
                    0xcb2da0dfbabcd69d,
                    0x2f7d840bc0e24ea6,
                    0x4718e687313e3531,
                    0x237e5593045bfcbd,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x4d63358e001e231a,
                    0x3b0bb13687eeb122,
                    0x884d78fc8f1a8d16,
                    0x1461e7fb666a4f66,
                ])),
                Felt::new(BigInteger256([
                    0x7bc7e7e8bc8864f4,
                    0xfe45be09e8932049,
                    0x8483a3075100f372,
                    0x1e237c02655f01ca,
                ])),
                Felt::new(BigInteger256([
                    0x3312f97ef92de2e6,
                    0xee4666fb4a9ae783,
                    0x0113a2fd5ec19fcc,
                    0x27d3c28abda285fa,
                ])),
                Felt::new(BigInteger256([
                    0x1778dd9db7d916e2,
                    0x76195814e4d31cb1,
                    0xc334e8eccb31648a,
                    0x22b1d7f96edca665,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xa5fc5b957385f80c,
                0x3e55b86c679491cd,
                0xb4164befd21ca60e,
                0x0df3d4513298dfa5,
            ]))],
            [Felt::new(BigInteger256([
                0xf2e09b850abb9a38,
                0x2de0c6944b244cc8,
                0x3e458ca92ba5ecdc,
                0x33d824bb93b724fe,
            ]))],
            [Felt::new(BigInteger256([
                0xa8a5a2304db2856d,
                0x1f52b30858452497,
                0x7a01a09cabedcd08,
                0x2dfb4da6a57069e3,
            ]))],
            [Felt::new(BigInteger256([
                0x59a4175858043d1f,
                0xd6401e5d201fcb8f,
                0x01062301f66afcc8,
                0x273fd73e140799f4,
            ]))],
            [Felt::new(BigInteger256([
                0xf1011877bcb199c7,
                0xc811fee187f0c92d,
                0x7cb73205b0a05411,
                0x10a94b94a289f280,
            ]))],
            [Felt::new(BigInteger256([
                0x9f40a5da7f385fbd,
                0xb308ed7241c58140,
                0xa37806d81451b267,
                0x2c8c3666e315c2a7,
            ]))],
            [Felt::new(BigInteger256([
                0x4850b0d60bfb52a9,
                0xcac60a6eff7c25d0,
                0x0cb9de809e094d73,
                0x2a330136a8e97385,
            ]))],
            [Felt::new(BigInteger256([
                0x7b35051d91a70d49,
                0xdfb7e0f27cd97316,
                0x729123ac1a9b497d,
                0x38351eda4b19d6e9,
            ]))],
            [Felt::new(BigInteger256([
                0xfb984d00e683dc61,
                0x2974f65d43225c3f,
                0x4046b698fb6ef9c6,
                0x04ed33f375a6a389,
            ]))],
            [Felt::new(BigInteger256([
                0xa667360b99d9b6ee,
                0xfa0c93db5fbaebe1,
                0x377c5a9e03caa35a,
                0x3606705fb2eb18c3,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
