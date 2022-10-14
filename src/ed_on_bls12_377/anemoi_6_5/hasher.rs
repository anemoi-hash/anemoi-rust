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
        // Generated from https://github.com/vesselinux/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 6],
            vec![Felt::one(); 6],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger256([
                0x488e042483636473,
                0x638c8ffaea115517,
                0x3384a3b15e5a95bb,
                0x029f19af14adf4c9,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x55a7e02a89d1a8ef,
                    0x4aef93817a6f6b78,
                    0xcc99c15b217e025b,
                    0x04206616c02fbe2c,
                ])),
                Felt::new(BigInteger256([
                    0x5134fcd4735de381,
                    0x682ef5e75eb77b38,
                    0x520c7f4d528a8f79,
                    0x0af2e8213a029a3d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xafbdcf38de14fe82,
                    0x50bd979029bc80c7,
                    0xf4eac0a2802591a7,
                    0x0c3be384fb6ca3e2,
                ])),
                Felt::new(BigInteger256([
                    0xeb45627fcc70e93c,
                    0x8382c07841d1e787,
                    0xf0b35dde9f3e93fc,
                    0x00e74316ce6b7b3e,
                ])),
                Felt::new(BigInteger256([
                    0xc35e50b0fd02d693,
                    0x821fef4c2dc968fd,
                    0x72a47a6e0fc18a3b,
                    0x113bcd2a6c28bacd,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x4e75f27e6821a21e,
                    0x9508313e011c5f10,
                    0x65c5f1a17b2ddf0e,
                    0x10b3700647ed04b8,
                ])),
                Felt::new(BigInteger256([
                    0x8bd89ed8128b2333,
                    0x882869d509450753,
                    0x815bbc6069366135,
                    0x01dd3c3fb0920956,
                ])),
                Felt::new(BigInteger256([
                    0x564d7710597d8d0c,
                    0xcb00ec19185a64d1,
                    0xe4f4b9752e630fea,
                    0x044577c56baa4ead,
                ])),
                Felt::new(BigInteger256([
                    0x31a0472e2d077179,
                    0xa6dbfa879afdd304,
                    0xd04de6c2bfeb9b4e,
                    0x0c7fc16cd715a061,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x584cb388c517b9ee,
                    0xff226f881aef1617,
                    0x933ef9339cff1a28,
                    0x0afd92074a84a2fb,
                ])),
                Felt::new(BigInteger256([
                    0x28897b4ee3591cc3,
                    0x7ef83dd628627624,
                    0x7d50f9fa126a684f,
                    0x0cc638bf6391ff86,
                ])),
                Felt::new(BigInteger256([
                    0x3849d70993c8c868,
                    0xfb73473dc4917941,
                    0x443a68224963065f,
                    0x05801938b23570af,
                ])),
                Felt::new(BigInteger256([
                    0x4641a979c51f24bd,
                    0xa287d3777a4fc8f0,
                    0x00d333389576c034,
                    0x0b1f797a749f5a20,
                ])),
                Felt::new(BigInteger256([
                    0x93f437ced6c37541,
                    0x52539ec305737540,
                    0x3e367ca899853bd5,
                    0x07fd7df9beb5498d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x56d110aee9e6d395,
                    0x0f308c71e7fac31d,
                    0x6ded3a93ecaf95f5,
                    0x0aca337c2ab39a8f,
                ])),
                Felt::new(BigInteger256([
                    0xfa21980022e0f3da,
                    0xe652c7d019f74dba,
                    0x6c8cba06ae327b82,
                    0x08d5c8d03c896f43,
                ])),
                Felt::new(BigInteger256([
                    0xe64fea2e6c7dd38b,
                    0x0f51f694ddd5953d,
                    0x5e711a3d9ddbc681,
                    0x0c73f91b35f424d3,
                ])),
                Felt::new(BigInteger256([
                    0xa5e3d880568bfb27,
                    0x8a493327ed27d6c9,
                    0xa49cbda19f05fabf,
                    0x0664b8dc7fd88ec9,
                ])),
                Felt::new(BigInteger256([
                    0x8c6e399ecd4840fa,
                    0xccc9915169e1e28c,
                    0x0ea25fdc38469b89,
                    0x0a835db4225a06bc,
                ])),
                Felt::new(BigInteger256([
                    0x96bbb7674b597b38,
                    0x7ecdef51567bbb32,
                    0x184b29152dde7974,
                    0x04a9e526575cb645,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x17a19fed4fff1b4d,
                0x029fb7132caf4f16,
                0xeba4551bec536b3b,
                0x1196f76d66011688,
            ]))],
            [Felt::new(BigInteger256([
                0x45b93ab493f1e444,
                0xe24462557d51934e,
                0xc4d1c3fe8d0c41cc,
                0x00fc7d0876a3616f,
            ]))],
            [Felt::new(BigInteger256([
                0xbe8bbc17b5b0bb2a,
                0x96b17564a1c966f6,
                0xb96784957a66e435,
                0x0134064da510ae08,
            ]))],
            [Felt::new(BigInteger256([
                0x60b0a0724d25072c,
                0x2266b94be3e71732,
                0xba7e731f436883b8,
                0x0f4b2e7b544c08c5,
            ]))],
            [Felt::new(BigInteger256([
                0x2608fff7c69e6b32,
                0x36bcc8524cde25b5,
                0xbd7c6cd18b3a6788,
                0x0e5be03d168dd9cd,
            ]))],
            [Felt::new(BigInteger256([
                0x8a365b5db5f3ad07,
                0xe31da3bd7ef491f7,
                0x88b3c6caa88d53c0,
                0x031be1544865d656,
            ]))],
            [Felt::new(BigInteger256([
                0xfbc79eaf9fe34a89,
                0x7bbb9e2716585940,
                0x1d3816869bed1d02,
                0x11c5ec4ed740b8f9,
            ]))],
            [Felt::new(BigInteger256([
                0xc1da32931e15ee6e,
                0x0736fa95e4b8e732,
                0x793b930a907ece6f,
                0x073dc1b82ce2ac69,
            ]))],
            [Felt::new(BigInteger256([
                0xc6dfb64cbb689ab8,
                0x201dc9596703b984,
                0xfbed328842683324,
                0x051bf67df2887cc4,
            ]))],
            [Felt::new(BigInteger256([
                0x9a67a76b347bfef7,
                0x2a4d3d9bfcefe720,
                0xe41fb91f99c4fd5d,
                0x1172034d6d981efb,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/vesselinux/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 6],
            vec![Felt::one(); 6],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x62e110d1dfd0a036,
                    0xd525ba744c82116d,
                    0x8d29b741673a793d,
                    0x0468052e63d6edc7,
                ])),
                Felt::new(BigInteger256([
                    0xafa4ab06ce8063ae,
                    0xe892881258f6a8f7,
                    0x70ab69e7b21b81ee,
                    0x0cfb181d0de0129e,
                ])),
                Felt::new(BigInteger256([
                    0xcdd04c628e739da0,
                    0x2f7975eccb0e106a,
                    0x281d7372fdff8220,
                    0x007aa60460cabd4d,
                ])),
                Felt::new(BigInteger256([
                    0xac1f9c7255044596,
                    0x53909da630d6d67c,
                    0xd3e943bdde8481a0,
                    0x0fc94941e1ab1e30,
                ])),
                Felt::new(BigInteger256([
                    0x99792938c209b4c8,
                    0xdf7cea4e356fdefd,
                    0x070852140075fb10,
                    0x0b9bca40a1d8b537,
                ])),
                Felt::new(BigInteger256([
                    0x96db1bb8c75027d8,
                    0xa1388c4e4f712d99,
                    0xfcfdae4bbdb6da2c,
                    0x0bf06c57d59907da,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0159d1f16c8702b5,
                    0x849e3259d3c2012e,
                    0xd4c3687ee5b7cdef,
                    0x09b2f01a0047e541,
                ])),
                Felt::new(BigInteger256([
                    0xc960d17e400e5235,
                    0xe939c3f782734ec7,
                    0x587dc458e50617cb,
                    0x094495d1220b0f4b,
                ])),
                Felt::new(BigInteger256([
                    0xe967e2517bb4cae0,
                    0x73313c70612b4013,
                    0xd8b9e306907ee786,
                    0x0d10f84b848ff433,
                ])),
                Felt::new(BigInteger256([
                    0xfe21eb817955908b,
                    0x5b0dc4ccfec139c9,
                    0x6fcad8d6fff39912,
                    0x0f4cb53ad5fc8b31,
                ])),
                Felt::new(BigInteger256([
                    0x59828460ab77f590,
                    0xc194bf387fa56572,
                    0x2c5e1e5adeb091be,
                    0x05a95a0d0377eb55,
                ])),
                Felt::new(BigInteger256([
                    0xc8f81df44de2a0ff,
                    0x8f76ffa8d1014e3e,
                    0x2d04b25d0b181eb6,
                    0x0e6cef18e2b11d8d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x04693b016d95c30b,
                    0x0f7ffdd8e2eed200,
                    0x0117dee7e7f52256,
                    0x10c18d8f52f13fb7,
                ])),
                Felt::new(BigInteger256([
                    0xcdbe4f2526645b91,
                    0x4a08644958089279,
                    0xf0ca5d20c5c9f8fa,
                    0x108376a2e66b6593,
                ])),
                Felt::new(BigInteger256([
                    0xf1fa426de1fe15c2,
                    0x4aa73e0a406ed463,
                    0xe81bf2aeaf635f94,
                    0x069074d94fd65104,
                ])),
                Felt::new(BigInteger256([
                    0x69b0557c616b38ff,
                    0xbc061975905b34d6,
                    0x8d6b90d95023f68d,
                    0x1014c486fce1083f,
                ])),
                Felt::new(BigInteger256([
                    0x241ac58c81bc9cfd,
                    0xbabd582cdb041a2a,
                    0x22fa85c4d93616e1,
                    0x0103622e362c9386,
                ])),
                Felt::new(BigInteger256([
                    0xf11af731aab075e0,
                    0xc2a73877cc3a0d41,
                    0x7e1d4816b8cb3acc,
                    0x0ca93700b56f080b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3bd9a4fa3d7352f3,
                    0x45432b08eea18b9f,
                    0xed92e809ac0c34d8,
                    0x0c1f1123c8d08664,
                ])),
                Felt::new(BigInteger256([
                    0x6f614f0d2abfcdd5,
                    0xb69ea125d9984ec5,
                    0x4dd879989e23ad11,
                    0x0784e381f3452f23,
                ])),
                Felt::new(BigInteger256([
                    0xfadadd705085820c,
                    0x3fd5c261e6c856b1,
                    0x3da16950b978c141,
                    0x0b51a5fd19be74cc,
                ])),
                Felt::new(BigInteger256([
                    0x9ea4dcadf2fdc4e1,
                    0x5cf78c15d4e603fd,
                    0x9e95fc675f062b25,
                    0x03939dd4c3dd8286,
                ])),
                Felt::new(BigInteger256([
                    0xe9df9511a8fa7c33,
                    0x5cd2b152d0dee07c,
                    0x53af83116aa6c43f,
                    0x0d9be84a5e9c4820,
                ])),
                Felt::new(BigInteger256([
                    0xefed5307dc6ad9e8,
                    0xded6f639d8fe10ed,
                    0x1da415402b457fac,
                    0x031c7de727d3a904,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7710ee86dea54d7a,
                    0xa336db1234affada,
                    0xf40ddf5e7994bfde,
                    0x045034282442e85c,
                ])),
                Felt::new(BigInteger256([
                    0x830d8acc1a2033e9,
                    0x41f170e4f4fe5a9d,
                    0xa26d5c1c133aa42c,
                    0x112dca84b8afdab7,
                ])),
                Felt::new(BigInteger256([
                    0xffb9fd6c3bd5b6ba,
                    0x163949b38fc62422,
                    0x26300bae74d79e1b,
                    0x0ca35e65ceb00617,
                ])),
                Felt::new(BigInteger256([
                    0xf6331f2807a417f9,
                    0x7a4ca38b47c3f317,
                    0xe3b054701ae782f2,
                    0x00c5ea2248521edd,
                ])),
                Felt::new(BigInteger256([
                    0x72a2f422dac5dea3,
                    0x05dc599d7bbe614a,
                    0xd9b99a5da774dfba,
                    0x062c72e21160a6d5,
                ])),
                Felt::new(BigInteger256([
                    0x942d700a3dc826ae,
                    0x3c9dfed1ef351f0a,
                    0xb95616a426eece4f,
                    0x096c55924a35b0de,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x20538d9b227ff965,
                    0x5310aacba473ff59,
                    0xd176262bc9800ab5,
                    0x05d42408e0db5a5e,
                ])),
                Felt::new(BigInteger256([
                    0xe1dfaf9ab5d8914a,
                    0xe06f82be26149700,
                    0x3b012ddfe337af84,
                    0x06f7d78565181296,
                ])),
                Felt::new(BigInteger256([
                    0xde3dec499cd5460c,
                    0x8bcf2100e1be2d12,
                    0x6414839f1fe209c1,
                    0x0d0b7406316daec4,
                ])),
                Felt::new(BigInteger256([
                    0x486b4a6b24647aa1,
                    0x8b28901e1fd6e315,
                    0xc665734da872c222,
                    0x02c0175496d278d9,
                ])),
                Felt::new(BigInteger256([
                    0xd217091d307fe7c1,
                    0x0c6d8307789a0ca4,
                    0xf3475d2f31f08ead,
                    0x0d42ac24758e1696,
                ])),
                Felt::new(BigInteger256([
                    0x0f35d0b6c3697e94,
                    0x461038cb67444d95,
                    0xa4f619c1a59b39f5,
                    0x107b516ce7c81ce9,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x1b9f0e9472f81015,
                    0x3c2339823bf1d66a,
                    0xeb3f1bf194bd98f5,
                    0x0416a84fbf9e76b5,
                ])),
                Felt::new(BigInteger256([
                    0x0ade0ac5426bbf55,
                    0x4bf0ce5770d03a68,
                    0x04540eaf9d6575d7,
                    0x05ae70d35745924f,
                ])),
                Felt::new(BigInteger256([
                    0xfe38f74f997f8669,
                    0xe5cda720abf23f31,
                    0x60622b82ef2c81f1,
                    0x0705e09c44c31eb3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe5321007fb56f97c,
                    0xccca2263f7a4e7b3,
                    0x2f22416b0efa31ab,
                    0x092a704386f5e7e6,
                ])),
                Felt::new(BigInteger256([
                    0xd58211e1af8552e9,
                    0x4f3f2ea76ee79a08,
                    0x444f7b72ddcc974d,
                    0x0f2e5e9d4e07d82e,
                ])),
                Felt::new(BigInteger256([
                    0x32c3c67ae779713b,
                    0xc7247deceab96ecd,
                    0x1c1b86d5dac30e1e,
                    0x057172c152133eef,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x51c68dc28b9c84b5,
                    0x90fff03c0bf5310b,
                    0x5941bda0fed29532,
                    0x0af7637f0250f9c9,
                ])),
                Felt::new(BigInteger256([
                    0x4ac7d67428ddd540,
                    0xe8f71d6c83c59ae1,
                    0x3fc891c9944e5663,
                    0x04977f81259a7729,
                ])),
                Felt::new(BigInteger256([
                    0xe82112bb992035a4,
                    0x425703911f93dd14,
                    0x8d8c825aae63ee49,
                    0x03b3e4a7bade3845,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x94adeec4c3027e5d,
                    0x6b673468fbde4fa1,
                    0x1f81f6d138df4410,
                    0x01e146a6773b7ccb,
                ])),
                Felt::new(BigInteger256([
                    0xca756f0406b0a5e7,
                    0xbe4cd917bfbeb821,
                    0xff646198da924b67,
                    0x02842c130c50bf29,
                ])),
                Felt::new(BigInteger256([
                    0x85aeaf16fa50b95f,
                    0xfc0643d1d95cd659,
                    0x8cf54eb736434f2e,
                    0x025c52422b941a64,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7674ef3a37562c98,
                    0xae7f51e22c8e473d,
                    0x076e1b1662e27ea8,
                    0x0d593fbf3e5db55d,
                ])),
                Felt::new(BigInteger256([
                    0x7a5e392bcea16f5f,
                    0x6ffb1be383ebb142,
                    0xc7763b2292a7d405,
                    0x0d9c16164a7b325b,
                ])),
                Felt::new(BigInteger256([
                    0x9a2d799ccc8d9534,
                    0x593e73e66ca9d835,
                    0x6a70b67b9c21e6fb,
                    0x0f0f62ed1633de74,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x80ab62de71a602d3,
                    0x18d0001a79cf1b6a,
                    0xf7513b4265d51458,
                    0x0bad7805e29c4283,
                ])),
                Felt::new(BigInteger256([
                    0x383c3bfff02a6b2b,
                    0xbf9fb18f90261621,
                    0x7305cdc2e9b0ed6e,
                    0x1080967fd0a665d8,
                ])),
                Felt::new(BigInteger256([
                    0xc492155b6132bd91,
                    0x2dae87e320bff301,
                    0x57765e46259efe67,
                    0x02f2ba4cb3d0aaad,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x58fc7384288f22ef,
                    0x9a4fadf1843bfa69,
                    0xa0918fceb8b7f330,
                    0x044cb2a35d20e61c,
                ])),
                Felt::new(BigInteger256([
                    0x73796d5064517786,
                    0xe17f797b2b0783e7,
                    0x7c00bb36554a337c,
                    0x0d0e04dc8a760912,
                ])),
                Felt::new(BigInteger256([
                    0xbe01c7737f10582f,
                    0x9703924ec9161cf7,
                    0x6a6494bd962e9347,
                    0x086cf61c1778f076,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x326b8514d6bdbefd,
                    0x3f9e8d3136230853,
                    0x57bf2c27c2738296,
                    0x00bc5d06298c0b76,
                ])),
                Felt::new(BigInteger256([
                    0x7cbbfef0460ea207,
                    0x07ac9dcac009019c,
                    0x31cb04f80d402904,
                    0x038d31f3d839de6e,
                ])),
                Felt::new(BigInteger256([
                    0x9d488062ef914bc7,
                    0x0cc964377c271452,
                    0x2fcd72711e7f4e32,
                    0x05a66472f7b5e25b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x78c2b83332d6ad6d,
                    0x4a451aa8eb104060,
                    0xac249979db08be3c,
                    0x0ab1e7c295ca8ec5,
                ])),
                Felt::new(BigInteger256([
                    0x54363ce498c2c7ab,
                    0x3385645fdb0a1789,
                    0x6a1524011483e0db,
                    0x0f5b1e7a1298db5c,
                ])),
                Felt::new(BigInteger256([
                    0xdb0522c2813bee2a,
                    0x9caa17ec028af17d,
                    0xc17584128eb68de8,
                    0x03572e85b30cf8b8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf5fa97ef7bab6138,
                    0xaaeda8f0605b29f0,
                    0x609c69a2fd349aaf,
                    0x0b99c4a5e3ce909b,
                ])),
                Felt::new(BigInteger256([
                    0xc58255cc6b3501d4,
                    0x9facc6c47babf016,
                    0x3b99d2fb58cc81c8,
                    0x0b1537dae9c2f781,
                ])),
                Felt::new(BigInteger256([
                    0x2a8c3ec2e9178547,
                    0xcc25559a48757c0d,
                    0x14c2fe1b986f0654,
                    0x01ae8dc7f53cc2c0,
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
            vec![Felt::zero(); 6],
            vec![Felt::one(); 6],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x4035d8d0d0957d2c,
                    0xb44c38494dee07e0,
                    0xd2fdafa08ce68241,
                    0x0942141fb39dbc6f,
                ])),
                Felt::new(BigInteger256([
                    0x065f6b924ab01800,
                    0x1f518d9731682c0f,
                    0xaae2b35953b384e5,
                    0x11e5055611ed9018,
                ])),
                Felt::new(BigInteger256([
                    0x5f40c28d30d962f4,
                    0xeef2fd8ab409ee5f,
                    0xd84c9011bfd4dfce,
                    0x022bda43cfb5063c,
                ])),
                Felt::new(BigInteger256([
                    0xf51955f313148f7b,
                    0x8e25ce7ca630adb6,
                    0xa3f3bc6232cb0424,
                    0x0d08b4f9786535a3,
                ])),
                Felt::new(BigInteger256([
                    0x0059e72e7b198f57,
                    0x7fdd3e6ccc2d9a06,
                    0xde8671317ae34689,
                    0x00fd4d94c9ae726e,
                ])),
                Felt::new(BigInteger256([
                    0xb46b58347d357ac4,
                    0xa9565765d9dca505,
                    0x00d7fed195927c92,
                    0x04b1ecabf2a70632,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7b9928d46a9f1bca,
                    0x021e808871c612af,
                    0x310c900f3a59eee3,
                    0x04674872487d364b,
                ])),
                Felt::new(BigInteger256([
                    0x4190d8d74dd3cb8e,
                    0xb717fcbbc7a592e3,
                    0x9fb0d44048657ffb,
                    0x037bca02dc680bae,
                ])),
                Felt::new(BigInteger256([
                    0x2672819221758cf0,
                    0x4b5ec0c97b2b3cda,
                    0x1fdbb4e2a8893f3f,
                    0x0eaa5f9b1a3cdc5b,
                ])),
                Felt::new(BigInteger256([
                    0x8fee29e8abe1237e,
                    0x38e2672ee4cd1005,
                    0x7bd3ded519485ef9,
                    0x114b09b24e17553d,
                ])),
                Felt::new(BigInteger256([
                    0x761fa753a9739e32,
                    0xbc81c7597cfc2a94,
                    0x16c68144777d2d11,
                    0x0121467773c2be93,
                ])),
                Felt::new(BigInteger256([
                    0xaac8ae5e220c46b7,
                    0xa7e27e3fb09a6210,
                    0xcdab4f5e847e8739,
                    0x02541655b3a65cad,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x253b92a3a7f1b558,
                    0x029f7f547007656d,
                    0x796c383f9ebe102b,
                    0x0e7f91633f678745,
                ])),
                Felt::new(BigInteger256([
                    0xc9f7b6a6e9ed3a69,
                    0xb15978a0ed1a7ce8,
                    0xb031cffca0bfc539,
                    0x08b18d5b57935cc8,
                ])),
                Felt::new(BigInteger256([
                    0xa7dd6926516d8b41,
                    0x5a737ac48d79a81d,
                    0x5aff250e0c7d23e4,
                    0x091011175795682b,
                ])),
                Felt::new(BigInteger256([
                    0xc38dd2c8b2a1a177,
                    0x1d6b2c3af5a64c98,
                    0x66cd724a1fbb3c25,
                    0x09a5dda8198fd0fe,
                ])),
                Felt::new(BigInteger256([
                    0xd4843e01d25111be,
                    0x2c7097dc4d3d2ab7,
                    0xe6d39168ca907c4c,
                    0x0d2fa2981b9b9467,
                ])),
                Felt::new(BigInteger256([
                    0x985232778e463af2,
                    0xc290c2c1aee788bc,
                    0x433339dc67faa46f,
                    0x0c6c568c8e527716,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x9ac0e445c770098d,
                    0xfc58d86a70901f7e,
                    0xe6f0d1d49da32142,
                    0x0c02cd8493e83df7,
                ])),
                Felt::new(BigInteger256([
                    0x572e1788626b7361,
                    0xe1607e0e327ea314,
                    0x1c35ea25ba0b221e,
                    0x0c117cd7f802cd56,
                ])),
                Felt::new(BigInteger256([
                    0x92049137bd8bf718,
                    0x400dc30b1a6812e9,
                    0x4ec2900c60217bc2,
                    0x0460429e86f2d470,
                ])),
                Felt::new(BigInteger256([
                    0x617266ea7a699ba5,
                    0x162c760dd5351cc1,
                    0xd78174f51fbfb34f,
                    0x0bd807d6192aac42,
                ])),
                Felt::new(BigInteger256([
                    0x30a40b17b2290e09,
                    0x4f2c31fbc117eb1a,
                    0x30319d3c8edaf029,
                    0x0a57e85cb8d85a44,
                ])),
                Felt::new(BigInteger256([
                    0x2dcbf6bb7a154198,
                    0x038feb7cb1ff57f2,
                    0xe83406f29ab6d1db,
                    0x07158d8fe81bd53f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x9fa4797f639c3b93,
                    0xa5687339283e7c57,
                    0xed1c779e28e414b0,
                    0x03051ccc0a53eb09,
                ])),
                Felt::new(BigInteger256([
                    0x1967ee11f26d25cb,
                    0x234d8ff0a418dab0,
                    0x9eb12dce05ec30bf,
                    0x0ddcb21bc5a042ed,
                ])),
                Felt::new(BigInteger256([
                    0xff521a8425faf0c3,
                    0xcd0b6150eda0d62a,
                    0x1014bb5b248f1221,
                    0x01748b4840848c69,
                ])),
                Felt::new(BigInteger256([
                    0x76bd3c0d7694b57b,
                    0x22eaddee693e439f,
                    0x65c5bd6a256e0bdc,
                    0x0cba17ee14910212,
                ])),
                Felt::new(BigInteger256([
                    0xcb20277c0afc4064,
                    0x54cade9734aae0b2,
                    0xdefe4fb2e23afb40,
                    0x0c0924c67e90619a,
                ])),
                Felt::new(BigInteger256([
                    0x7bf4ff1ac38c6b0f,
                    0xad6347151af9b2a6,
                    0x416af56d9ffa4e2c,
                    0x01bae45c2ead662c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x1ad45d1c3e1af36f,
                    0x189fc2b417ed35a0,
                    0x4cd2f95d2a02797a,
                    0x0a10702d9dcc882e,
                ])),
                Felt::new(BigInteger256([
                    0x0953096eaac83530,
                    0x91059f2353e91822,
                    0x69fe6b30d0091eb9,
                    0x060292fc2f2ece26,
                ])),
                Felt::new(BigInteger256([
                    0xb8d2d6ed547da643,
                    0x4fc916fc08189d39,
                    0x2d27d75f14766d40,
                    0x0c837b25d6e7d869,
                ])),
                Felt::new(BigInteger256([
                    0x850a3c2b69b234c4,
                    0xc2ee49260f775420,
                    0xeeb382f455d23493,
                    0x0e8700118637da92,
                ])),
                Felt::new(BigInteger256([
                    0x06e73c6e616fcabf,
                    0x99545a297f430189,
                    0x06055781052f4111,
                    0x0cc9dfbf300f19f0,
                ])),
                Felt::new(BigInteger256([
                    0x9beb73eed1d3b6a8,
                    0x16110b2bcad831af,
                    0xd14e544d671c661c,
                    0x121aa5ee97a498a1,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x24b610a94ee355d3,
                0x6de1aefa58b45004,
                0x4ff55624214f90be,
                0x10caf9bf5ba727b8,
            ]))],
            [Felt::new(BigInteger256([
                0xe36668649255bd9f,
                0x898357f98145f088,
                0x2ed8f6956b522716,
                0x0b1edc438ce459ad,
            ]))],
            [Felt::new(BigInteger256([
                0x7a9df6f24d9a8f98,
                0x62a39a3adf4ea900,
                0xc5e284a6e54d29de,
                0x00976249489d03e1,
            ]))],
            [Felt::new(BigInteger256([
                0xe4d20cdfc403dda3,
                0x25ba515294f9de1c,
                0xabdba72149b4dea7,
                0x06c1c4fbaf205659,
            ]))],
            [Felt::new(BigInteger256([
                0xd3f4d667cede510c,
                0xd4a4ed4a84595d1d,
                0xe0df870a1e2f9e3d,
                0x06043d536728db26,
            ]))],
            [Felt::new(BigInteger256([
                0xc2f7f89a9434f627,
                0x07c3f482a27762bc,
                0xfc3945aa115c45d5,
                0x06b3503279ee015c,
            ]))],
            [Felt::new(BigInteger256([
                0xf9e89741e4a39518,
                0x7268d56305eeda3b,
                0x4b9d94550d9fe0f1,
                0x0f5c2740b7e08382,
            ]))],
            [Felt::new(BigInteger256([
                0xd337a8cc193da772,
                0xf36f91adb0b2134c,
                0x265a1f0473ffccb2,
                0x005a60f7517befe8,
            ]))],
            [Felt::new(BigInteger256([
                0x33c422f80a2bc979,
                0x991665d76972f29e,
                0x75eed5b5109d133b,
                0x081816a1f4c94b8e,
            ]))],
            [Felt::new(BigInteger256([
                0xfb45b4462c0b601e,
                0xabc46387b4e4217a,
                0xccfbff97bd666ceb,
                0x0b87ad7a4e942b3b,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 6));
        }
    }
}
