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
    use super::super::BigInteger256;
    use super::*;
    use ark_ff::to_bytes;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
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
            vec![Felt::new(BigInteger256([
                0x13918ba1fdeb2e75,
                0x5c4636ff1d9522ce,
                0xe68e8887d0e29d65,
                0x11857e413d13ab83,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x8c8d02476c9fa916,
                    0xc5772c4f17765ec6,
                    0xb938071952cae86a,
                    0x0ee543c83a23dca1,
                ])),
                Felt::new(BigInteger256([
                    0x4a25dd4df9e75987,
                    0x195a6a5da4286efe,
                    0x247eda7c22e87784,
                    0x33b5ac890acc3bd2,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x89d84bde4ed8653a,
                    0x501476bde4a8226a,
                    0x46923abc3e7c943c,
                    0x1712d9424e3cf683,
                ])),
                Felt::new(BigInteger256([
                    0xc8ff2a820215fea0,
                    0x8474348807f42ac7,
                    0x80657494cadcb31a,
                    0x337d4cc40952018d,
                ])),
                Felt::new(BigInteger256([
                    0x42a903a2f1bc93c9,
                    0xbf952706f6510842,
                    0x0b60eb62955c8aa3,
                    0x07e6cba6db616ae4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5178b55479f4715d,
                    0xf02246c0714f04c2,
                    0x80193bb67625ba4c,
                    0x223d3a6d1bd4801c,
                ])),
                Felt::new(BigInteger256([
                    0xeb5d69cbec892ac6,
                    0x167c33eac3e6cfaa,
                    0xe8b2598645485162,
                    0x2853e5bc044bcdc5,
                ])),
                Felt::new(BigInteger256([
                    0x693b51435d1f19db,
                    0xea2a4acb814de394,
                    0x9b2dd67935a23373,
                    0x0033bc82dd4bd5b8,
                ])),
                Felt::new(BigInteger256([
                    0xd417cc00125ba5de,
                    0x22e9e0850533a7a8,
                    0xfb70dd5438d09069,
                    0x03fac41f49e81ce9,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xdbcb540b9680b370,
                    0x8a1e5b0a2bc0b673,
                    0x87ace093ceebbe16,
                    0x1af29c6e4a639ae9,
                ])),
                Felt::new(BigInteger256([
                    0x6582961f8916a5d8,
                    0x42f99ec6cbceb881,
                    0xd9d525a79ffe8b15,
                    0x3680363da192acc7,
                ])),
                Felt::new(BigInteger256([
                    0x7c745fbe941b318c,
                    0xb2ff7f9751e5c4c9,
                    0x0185ed93b04ec67d,
                    0x2b48d23ad6faab81,
                ])),
                Felt::new(BigInteger256([
                    0x84618ce56253b911,
                    0xc10b358b2a38a676,
                    0x57af5a6245b369eb,
                    0x24e179e3b189c01c,
                ])),
                Felt::new(BigInteger256([
                    0x689d4b1a9c96cab8,
                    0x0bcae91c3e17aea1,
                    0xcd2258a96294ce74,
                    0x0401eccba587af5f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xb06b69176fb0a1a1,
                    0x37f8381e45479e58,
                    0x5c8107db668e51eb,
                    0x00234c780a8077f2,
                ])),
                Felt::new(BigInteger256([
                    0x78f25d80666709a6,
                    0x115c5af4478e6b51,
                    0xc5312d01ecc93b6a,
                    0x13543af1a0fafdc3,
                ])),
                Felt::new(BigInteger256([
                    0x1d53603b1c4c0172,
                    0x84745d05158476f2,
                    0x9dddbe428259d994,
                    0x00cb040d74b8d0ba,
                ])),
                Felt::new(BigInteger256([
                    0xd99ec0231a8ac9d1,
                    0xc7bfaa10c7ec30b9,
                    0xea2c7cc1f6ceb42a,
                    0x20afdb3cd5ee842b,
                ])),
                Felt::new(BigInteger256([
                    0x1db4c5f3dc2d3881,
                    0xafc6e9f3b3432a3e,
                    0xc73a9be8963d2fa7,
                    0x0696253da31cba1b,
                ])),
                Felt::new(BigInteger256([
                    0x0ec66f4c2ed2136f,
                    0xc6f06913a31961a2,
                    0x05c642075eb00ad1,
                    0x1795602677825d6d,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x6dbc241fa4ddcb4e,
                0x668d9fe1e33cd871,
                0x4c780bc1f7dedc67,
                0x0cabb8cc85114886,
            ]))],
            [Felt::new(BigInteger256([
                0x37a0ecab5df93d03,
                0xbf17b247d1b8a37a,
                0x035c63d97e182b39,
                0x185a03a83df06730,
            ]))],
            [Felt::new(BigInteger256([
                0xfa029e36936a9033,
                0xaaf95c31cb7b8333,
                0x7539e49dc7e26fd5,
                0x394477eb8eed12e2,
            ]))],
            [Felt::new(BigInteger256([
                0xeca924377e63a062,
                0xf24e1ee71e2f56ac,
                0x6ca612b93f9df78a,
                0x2eff8aeab4845272,
            ]))],
            [Felt::new(BigInteger256([
                0x8569bec4020fab43,
                0xe1c148136b502b13,
                0xf4a1fec8133f7061,
                0x07f4b2752926b5ae,
            ]))],
            [Felt::new(BigInteger256([
                0x50552575a8e00fa5,
                0x7f40b53429ef895e,
                0x2a93a9ffe2266faf,
                0x26c0839b945f6331,
            ]))],
            [Felt::new(BigInteger256([
                0x2518f29064b32119,
                0xe595b3b560785fe7,
                0xcded619b7a8e1f2b,
                0x0bd857ebe1323b46,
            ]))],
            [Felt::new(BigInteger256([
                0xefe8e4c08e86af7e,
                0x29f46c69aa68340c,
                0xf5d8810f5d92416f,
                0x2b0866d89b118fb7,
            ]))],
            [Felt::new(BigInteger256([
                0x6d9d8db3b29b0874,
                0x6f14e927100403f3,
                0x9301c2281849b08a,
                0x1f1c1516e75620fa,
            ]))],
            [Felt::new(BigInteger256([
                0x03713930e388b90b,
                0x2e8858ef17e73172,
                0xc0b707a589f7f02d,
                0x06a5c17c1e6bbf32,
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
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x6dbc241fa4ddcb4e,
                0x668d9fe1e33cd871,
                0x4c780bc1f7dedc67,
                0x0cabb8cc85114886,
            ]))],
            [Felt::new(BigInteger256([
                0x37a0ecab5df93d03,
                0xbf17b247d1b8a37a,
                0x035c63d97e182b39,
                0x185a03a83df06730,
            ]))],
            [Felt::new(BigInteger256([
                0xfa029e36936a9033,
                0xaaf95c31cb7b8333,
                0x7539e49dc7e26fd5,
                0x394477eb8eed12e2,
            ]))],
            [Felt::new(BigInteger256([
                0xeca924377e63a062,
                0xf24e1ee71e2f56ac,
                0x6ca612b93f9df78a,
                0x2eff8aeab4845272,
            ]))],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 248];
            bytes[0..31].copy_from_slice(&to_bytes!(input[0]).unwrap()[0..31]);
            bytes[31..62].copy_from_slice(&to_bytes!(input[1]).unwrap()[0..31]);
            bytes[62..93].copy_from_slice(&to_bytes!(input[2]).unwrap()[0..31]);
            bytes[93..124].copy_from_slice(&to_bytes!(input[3]).unwrap()[0..31]);
            bytes[124..155].copy_from_slice(&to_bytes!(input[4]).unwrap()[0..31]);
            bytes[155..186].copy_from_slice(&to_bytes!(input[5]).unwrap()[0..31]);
            bytes[186..217].copy_from_slice(&to_bytes!(input[6]).unwrap()[0..31]);
            bytes[217..248].copy_from_slice(&to_bytes!(input[7]).unwrap()[0..31]);

            assert_eq!(expected, AnemoiHash::hash(&bytes).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
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
                    0x2269783d1714e2c8,
                    0x097a414dad6c21a4,
                    0x74f661ca2a8556da,
                    0x0aaa259c982c6e81,
                ])),
                Felt::new(BigInteger256([
                    0x99a3053accb08890,
                    0x91127fdeb7b6115b,
                    0x4fd77b23576173d1,
                    0x361235704c82722e,
                ])),
                Felt::new(BigInteger256([
                    0x24a9fa9ef4643f86,
                    0xeaab12b36f4a7990,
                    0x53ee864900e60596,
                    0x318b5ecc0aaa8f35,
                ])),
                Felt::new(BigInteger256([
                    0xeb64e63ff9473ec4,
                    0x6cabe22f54fd56ad,
                    0xbfdb66d4b4b54291,
                    0x222b5f43c2b42a2a,
                ])),
                Felt::new(BigInteger256([
                    0xe7aced64a9865bb3,
                    0xe3c2f0825a7f35a1,
                    0x5e9803da88389c2c,
                    0x16194080fab01708,
                ])),
                Felt::new(BigInteger256([
                    0xb212c239e9c6bd9a,
                    0x12bdcb47a0890e42,
                    0xf3e4de2b0b9ea881,
                    0x0eaa1caef0effa0e,
                ])),
                Felt::new(BigInteger256([
                    0x98d56b8c7a713541,
                    0x279833245efdc007,
                    0xcf461f5af493c326,
                    0x0cd9f771febb32b3,
                ])),
                Felt::new(BigInteger256([
                    0x1ac19fe17833b6f7,
                    0xe055a6c462f5c2cc,
                    0x154b9f37b0c65de0,
                    0x02bcf004c6268164,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x13a2315eefc65a63,
                    0x30c72fc7ed4c50c5,
                    0xb1605dc7e76abd51,
                    0x242b2bba817e0485,
                ])),
                Felt::new(BigInteger256([
                    0x3ffc0524e2a45e8a,
                    0x7b4a7a90efc662af,
                    0x7ce42ed094079fbe,
                    0x124b8a706a173099,
                ])),
                Felt::new(BigInteger256([
                    0xecaba59fa72378aa,
                    0xdd528f0a57ecc91a,
                    0x3e75705b25bd1b86,
                    0x06c5a629e022085c,
                ])),
                Felt::new(BigInteger256([
                    0x6cd5dbc2b8fcb8d8,
                    0xbaf05db388945af2,
                    0x05022cdfcb007625,
                    0x12c7abadc80e3403,
                ])),
                Felt::new(BigInteger256([
                    0x5f6256447a6b928d,
                    0xb3225efbe28c4a54,
                    0xfe938ba39c617d50,
                    0x0234b6ef2d240e35,
                ])),
                Felt::new(BigInteger256([
                    0x0c9f93d2ef11eb22,
                    0x66f6582047662af0,
                    0xb327c4ccdec02b1e,
                    0x11da3cdf5cbfb93d,
                ])),
                Felt::new(BigInteger256([
                    0xe4ab715216a7686d,
                    0x4cb94b4711e537dd,
                    0x07db5556eec34233,
                    0x298cf5cd0121d8cc,
                ])),
                Felt::new(BigInteger256([
                    0x5c091d3cb562c8cc,
                    0x86bf71d0683021c9,
                    0xaadc623e5f1b1924,
                    0x29e661b1a9b29620,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xbe5ca3ce0c31279e,
                    0x928e83363e713b30,
                    0xe56f0493ee647151,
                    0x24c92bc0ef3f708d,
                ])),
                Felt::new(BigInteger256([
                    0x2e0ff764c0c340d6,
                    0xde256889b6ecc492,
                    0xebc0ef62c2b7b8fd,
                    0x04de03481fc37608,
                ])),
                Felt::new(BigInteger256([
                    0x6f79ad06792d5f67,
                    0x888f87c612633bb8,
                    0xfbc5a9416c5d4b69,
                    0x20384e1cbae20486,
                ])),
                Felt::new(BigInteger256([
                    0x46a87edee4f7c12f,
                    0x2d4a97f5f1cbf052,
                    0xc51d86c719d85490,
                    0x143dc1c9db43d6a8,
                ])),
                Felt::new(BigInteger256([
                    0xbc74ea17803ae1d4,
                    0x850aa29cd30a932d,
                    0x617f7cdb37df4951,
                    0x369a55c0bd51b804,
                ])),
                Felt::new(BigInteger256([
                    0x3d1ad17cba263fbe,
                    0xafb46ccf5471c679,
                    0xb15a4cf82853d2c4,
                    0x2f6e59c6a64932c2,
                ])),
                Felt::new(BigInteger256([
                    0x71a2be335a70f53d,
                    0xebfc57e2b20eb9b5,
                    0x7829777577a1c0b4,
                    0x063a293a1f4c703f,
                ])),
                Felt::new(BigInteger256([
                    0xaa4ec54885d97c0b,
                    0x30bf73fde536fc83,
                    0x78e4ccd5a5208c8c,
                    0x21d407a387c1c70a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x8cb486245b2893f5,
                    0xaf7ee9ffde991b65,
                    0xdd693d5c2b970376,
                    0x3f9e5636d857a4f7,
                ])),
                Felt::new(BigInteger256([
                    0xde6833773cccba32,
                    0xd2ab269ecc348c61,
                    0x0af27d15ee2162a6,
                    0x31a148c80514f7ce,
                ])),
                Felt::new(BigInteger256([
                    0xd79da8db1cf8d274,
                    0x02bed6c58d75c29b,
                    0x3677aee0892d466f,
                    0x2457696d8862721b,
                ])),
                Felt::new(BigInteger256([
                    0x660aa3566a85e6b3,
                    0xd0d7ecffb846c919,
                    0x9698fe4772e53e74,
                    0x05194628498d6847,
                ])),
                Felt::new(BigInteger256([
                    0x51ec9add90a37636,
                    0xa37ff4424f0e42d8,
                    0xcb62e01c9e223609,
                    0x0c0454627dc5c490,
                ])),
                Felt::new(BigInteger256([
                    0x88f9a4f89a57ffbb,
                    0xee8a4be51215d6b7,
                    0xd973fea33a8ea8a9,
                    0x232703b621b44760,
                ])),
                Felt::new(BigInteger256([
                    0x889166d8017c6145,
                    0xe31f0fe9f9135995,
                    0xaac562fc55f24f95,
                    0x109484b93de72943,
                ])),
                Felt::new(BigInteger256([
                    0x48807e5d0955d28e,
                    0x22d9fc6a9a193031,
                    0xa9f0f1e8e8f4c117,
                    0x05a2925a237194eb,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xfa4858984ebad484,
                    0x09db5adee28b3298,
                    0xceed3e2bf646aaee,
                    0x1bbf483d67b770cb,
                ])),
                Felt::new(BigInteger256([
                    0xb00030d1d8989add,
                    0xb2a2bcd3c687d582,
                    0xbba2fcf149807c44,
                    0x36f98d59ae33fafe,
                ])),
                Felt::new(BigInteger256([
                    0x7859fecda02050ec,
                    0x024fb21198fa94d0,
                    0x62653098c5036888,
                    0x15489cbc00c89c3d,
                ])),
                Felt::new(BigInteger256([
                    0x48abb4c51cf2a18d,
                    0xc1bebcf3a8e8cad6,
                    0xe3742e921365a8cf,
                    0x378561dae48f0a76,
                ])),
                Felt::new(BigInteger256([
                    0x5fbdae58830230f8,
                    0xff150f5725c8d193,
                    0xafa5b8ab1c66c251,
                    0x3c260ca74a8dc40f,
                ])),
                Felt::new(BigInteger256([
                    0xd59c2199e63c59fb,
                    0x6831c4c06ee3e054,
                    0xecbdfb2d015683b0,
                    0x1db147d514098ee0,
                ])),
                Felt::new(BigInteger256([
                    0x8fb23dabe7c67c43,
                    0x94a57faeb3f90e8a,
                    0x77fbe9e4dbf5360a,
                    0x3340d69956a5ee38,
                ])),
                Felt::new(BigInteger256([
                    0xd21fb034d201f5b5,
                    0xa92f253b09b002ff,
                    0x8b4f36f925c6c8df,
                    0x1d74057a74add857,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0fce4f55eb9778db,
                    0xa3930c0afaf03059,
                    0x850783f21c78adfa,
                    0x2dcc3dd9f3fc821f,
                ])),
                Felt::new(BigInteger256([
                    0x1e104a0e39f14327,
                    0xa4bf1dd1404eb93c,
                    0xff15b7dac68bb11c,
                    0x1857c99e0b4c60d6,
                ])),
                Felt::new(BigInteger256([
                    0x204b678ce2df069f,
                    0x4d01bcd1b3f8a902,
                    0x0c0f55f86862074f,
                    0x21b0ee885790766b,
                ])),
                Felt::new(BigInteger256([
                    0xdc3a8a4ff5180882,
                    0xb8e7ff1f1748c9c3,
                    0x4462fd8bf6ac4e0e,
                    0x283a0378b1ec89ec,
                ])),
                Felt::new(BigInteger256([
                    0xf25fa5b37eb45124,
                    0xf14a76e0e10e0672,
                    0x290760a6b1d409da,
                    0x046cb19e85a90b23,
                ])),
                Felt::new(BigInteger256([
                    0xd8db586c687bffce,
                    0x39cf8c3c0af1cc38,
                    0xf0350a9c510158af,
                    0x3cd6cd7afac1049c,
                ])),
                Felt::new(BigInteger256([
                    0x3da16d55c9f38918,
                    0x6c4cb3494dd9d5cd,
                    0x121ce71f361db304,
                    0x394433032f325701,
                ])),
                Felt::new(BigInteger256([
                    0x8a83c87506a830b2,
                    0xfb70d0d8aeff3931,
                    0x1cddbee8ac52600a,
                    0x1ca8f7bd71a09995,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xd975066f634f9fc3,
                    0x032a47635613bbd9,
                    0x6924d2df8ef34f70,
                    0x3c0ab0824a6445cd,
                ])),
                Felt::new(BigInteger256([
                    0xe528f7445f0551a0,
                    0x03fda35856ffbdd1,
                    0xeba46842fd87cafc,
                    0x12b142616e66346e,
                ])),
                Felt::new(BigInteger256([
                    0xac569c75e275c44b,
                    0x7e5ec7581c7b8f49,
                    0x3f2c8d3f1a178a09,
                    0x1b65ef809db4dcb6,
                ])),
                Felt::new(BigInteger256([
                    0x29e58d9b4ed2bc15,
                    0xd7ccc718dce70878,
                    0x1935a3b840ead464,
                    0x259340e0e0d44e3a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf93c75cfa21c190b,
                    0x7fc2b6c84babe515,
                    0x90b9e69ef6a81cc0,
                    0x0496850be06aa663,
                ])),
                Felt::new(BigInteger256([
                    0x13bbda9f0239d968,
                    0xe9cf2c455391ed96,
                    0x20f68d7097474f9b,
                    0x2db5388bd92fa8c2,
                ])),
                Felt::new(BigInteger256([
                    0x6edfb5775b738927,
                    0x3e59c6a807b42902,
                    0xaa3ed89af4ba6f94,
                    0x11c471b0586b38e5,
                ])),
                Felt::new(BigInteger256([
                    0xaf7935f1661134c9,
                    0x1426fd499c7929af,
                    0x25d6a698dd2e8c40,
                    0x1a14a18cf3c1ea87,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3b04b8cea5e9a4c5,
                    0xd32412af347e82a4,
                    0x43623b3aae74d011,
                    0x19f100d26505f5a2,
                ])),
                Felt::new(BigInteger256([
                    0x2f25ab4b691c02c1,
                    0x84762f62aca03884,
                    0xeccf1a303a712dcd,
                    0x1e4beef0b23a4f42,
                ])),
                Felt::new(BigInteger256([
                    0x9acb4a11057b5291,
                    0xcddc17123a765012,
                    0x4f1ad21801055be5,
                    0x3599d5298ccd0def,
                ])),
                Felt::new(BigInteger256([
                    0x2a42ff52dfcfa09e,
                    0xf7b83234d439572d,
                    0x327ddbeb8a04f049,
                    0x0c32a4d49cf900d1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9a0fbf869dce0257,
                    0xcc5a98eaa36b6bcb,
                    0x4f6b2e01cd50879e,
                    0x2f0c349614c8cd00,
                ])),
                Felt::new(BigInteger256([
                    0xd39168a91489b6a8,
                    0xeda97eb983f49c95,
                    0xdbd83c02b1de386f,
                    0x27c635f601c1a7a1,
                ])),
                Felt::new(BigInteger256([
                    0x1a2dafd1ceff0c7b,
                    0x3731aaf09f4af053,
                    0x6b48f519ce84699b,
                    0x2c1a4906af704f50,
                ])),
                Felt::new(BigInteger256([
                    0xf315d0a05fb83b98,
                    0x908cb5e3e8783ca8,
                    0xfb8a1452261d7a8f,
                    0x3cb607cba3fefcd9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x828a203c230f2da0,
                    0x4a7cddf27614ea95,
                    0xb9b793faee150533,
                    0x2c0ad3b4799ab38f,
                ])),
                Felt::new(BigInteger256([
                    0xf7dc90bc2ac13159,
                    0x99ebf45bc101a314,
                    0x318bdc673a203f8e,
                    0x361be3c26066c0a8,
                ])),
                Felt::new(BigInteger256([
                    0x2d6f5f644b51f373,
                    0x011a2cf6f6d05f1f,
                    0xc732ba4dc7369f42,
                    0x0df032d5e02f3bdc,
                ])),
                Felt::new(BigInteger256([
                    0x83d8bfc8071a6060,
                    0x14539f77a911da8a,
                    0x26943c7f177a297e,
                    0x29e7d89597e56810,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x184a3e6e1c7e4b5c,
                    0x015701ce278cfc9a,
                    0xf9e2ac14f96e5b85,
                    0x3884ec6267c945fd,
                ])),
                Felt::new(BigInteger256([
                    0xd7f1f5ab101a787b,
                    0x41d00b65d1979403,
                    0x1d292e3f52a6d82c,
                    0x1d0262072527f87e,
                ])),
                Felt::new(BigInteger256([
                    0xb2de7543e961f1aa,
                    0x20e0935a911e7e2d,
                    0xbf960ca7e6c19761,
                    0x3df2436b049e1124,
                ])),
                Felt::new(BigInteger256([
                    0x87e118366afd0314,
                    0x9aa4addc2a65b99a,
                    0x916c98465ee9b9ae,
                    0x3ee3bb2217dac797,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xefb741dd64e75a2c,
                    0x4432c5b7826064bb,
                    0x336e54b03ef5fe4f,
                    0x29bd436d58437654,
                ])),
                Felt::new(BigInteger256([
                    0x8fd3e4527e38c461,
                    0x942bd3b26238d67d,
                    0xba30eff913eff55c,
                    0x3111600e9b5f4ad8,
                ])),
                Felt::new(BigInteger256([
                    0x03971e808cdd2e3e,
                    0x22292a7e155d1a94,
                    0x00a1f3200e8f42fe,
                    0x2d08c9db0b86b4d5,
                ])),
                Felt::new(BigInteger256([
                    0xa76d872197f1745a,
                    0x2b2a511f264edc3c,
                    0x641d2f8ce060dce3,
                    0x1557a47c1a850768,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x25df8b49673e577a,
                    0x0d0f8b320ec1ec8e,
                    0xb943101925b73257,
                    0x1315c836b4f864d2,
                ])),
                Felt::new(BigInteger256([
                    0x9aa262ea3999b3a0,
                    0x59488fa47f3f8e1b,
                    0x80ac0b6abf10b8e3,
                    0x298192657b83a7f3,
                ])),
                Felt::new(BigInteger256([
                    0x3e4ab521cca82a1e,
                    0x390eee805066e443,
                    0xd2a212960b7a6ffb,
                    0x297d5b0f2ae72a97,
                ])),
                Felt::new(BigInteger256([
                    0x0db2a439c2e37769,
                    0xb52c61f40143aedd,
                    0x6a54d5e4ade0a42b,
                    0x036c62b8fd0eadbb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5a99aee0a54c61f4,
                    0x28bd368a6306257b,
                    0x048d60f5d1bb083b,
                    0x2f7661179d93fb4d,
                ])),
                Felt::new(BigInteger256([
                    0x8751b5c54023619d,
                    0x5141cd241a5c063c,
                    0x301614c1f35deb83,
                    0x3d50238256e4a303,
                ])),
                Felt::new(BigInteger256([
                    0xaac7d2508725aee1,
                    0xa639a4c2ae5f42d3,
                    0xa43ac7d052d3273c,
                    0x2ffddfc8e704b992,
                ])),
                Felt::new(BigInteger256([
                    0x4030b3db453d7561,
                    0x50734b4d946b0fd9,
                    0x67b0a25ce6125c8c,
                    0x36bdfeb6c256f596,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xce3bb65e47a65c13,
                    0x8851822402d778d4,
                    0x151af43441e559f5,
                    0x21446fce5e4da946,
                ])),
                Felt::new(BigInteger256([
                    0x3373fd509343027e,
                    0x7f38fcddfd4eb19e,
                    0x4fa5b10d401c76ae,
                    0x0241b57041990e56,
                ])),
                Felt::new(BigInteger256([
                    0x9fd6a45ecff16948,
                    0x4333d91fb9b614ba,
                    0xb12d6ca9bff47f05,
                    0x061b86730d0d299f,
                ])),
                Felt::new(BigInteger256([
                    0xd4d4c8ed96acb7cd,
                    0x1335789f97d48e53,
                    0x5278e33249712ae9,
                    0x1d8353d3f51ecf66,
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
                    0xb0c820caf54ec713,
                    0x39e2757ea0a7b749,
                    0x471314fd3d890bae,
                    0x099b24b19b3f55a4,
                ])),
                Felt::new(BigInteger256([
                    0xca0d5154193139e1,
                    0x6e00d4afa967a3a4,
                    0x0627701d13b111df,
                    0x2c0cf4433ff48aec,
                ])),
                Felt::new(BigInteger256([
                    0x584037bb46fcb5d1,
                    0x976986e813a946e8,
                    0xdad600cf75ea9cfa,
                    0x08bd41195b213d52,
                ])),
                Felt::new(BigInteger256([
                    0xa7c7826b6cd2a683,
                    0xf846eed4cc3cc5a8,
                    0xb539d57709f60ad1,
                    0x3351d3bb54050161,
                ])),
                Felt::new(BigInteger256([
                    0x0375327b156b1894,
                    0x8a68bea662db4264,
                    0xf89e8b988a4a7e92,
                    0x042c7645f2cd39bd,
                ])),
                Felt::new(BigInteger256([
                    0x56c5b43e577f0eda,
                    0x05f55e40f7b8f968,
                    0x951bfe33d81de7ce,
                    0x05611e2cba693924,
                ])),
                Felt::new(BigInteger256([
                    0xc15467fb45be7d48,
                    0xb9e356bbfd536571,
                    0x12e738e1aad47f33,
                    0x146e1518060c6005,
                ])),
                Felt::new(BigInteger256([
                    0x70ae0820960b7e6e,
                    0x81408f48562bc411,
                    0x899529e87692c651,
                    0x0124b735017e86ba,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2aa513261c2e9287,
                    0xb3c473c38af5b186,
                    0x11810ab107043b03,
                    0x1300221b9ed828b9,
                ])),
                Felt::new(BigInteger256([
                    0xd151b01e2c6725c1,
                    0x3d655cc306e4be11,
                    0x035966c977d02ceb,
                    0x3aba40b4dcb6b271,
                ])),
                Felt::new(BigInteger256([
                    0xd04d0135774151dd,
                    0x3ae2220513861cf9,
                    0x7470e6a6c89cf9cf,
                    0x2d8846920a0d51d3,
                ])),
                Felt::new(BigInteger256([
                    0xb6a4568e5f4da511,
                    0x40cab0df6982d199,
                    0x392c651fd316f1fb,
                    0x3f4e8d6609a741f2,
                ])),
                Felt::new(BigInteger256([
                    0x678ce9d3667bd04b,
                    0x8cff53052f3c23c9,
                    0x4a6ad622beeac2ef,
                    0x1a55e462da723cb0,
                ])),
                Felt::new(BigInteger256([
                    0x5af0cbd78ea67234,
                    0xc502fc9636df69e0,
                    0xe77cc68f1b5f3d07,
                    0x0420b18777d0e355,
                ])),
                Felt::new(BigInteger256([
                    0xa7f7562e34df5bcd,
                    0x66ba265911414299,
                    0xd813e3303fc15d65,
                    0x1feb842306eef6e3,
                ])),
                Felt::new(BigInteger256([
                    0xfb29cab2dc1cecdf,
                    0x7104b204736c62da,
                    0xb1ed166f90c25419,
                    0x097d37d96afabbbb,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5cee7bd1eeb142ce,
                    0x04c2764610dc3d13,
                    0x5bfafdb623e920c6,
                    0x3e8b6613848bf89b,
                ])),
                Felt::new(BigInteger256([
                    0x709b8732b329816f,
                    0x39c37804f1dfd407,
                    0xb5c00aa0ac9e0f97,
                    0x0ad64177e9bf6a89,
                ])),
                Felt::new(BigInteger256([
                    0x8f759987e0e7bd9d,
                    0x97acb482bcbc13af,
                    0xe429b772bd3da7e0,
                    0x00ba4e51b62fc8b6,
                ])),
                Felt::new(BigInteger256([
                    0xa3cf09016f91728e,
                    0x276e45b4348ee2fe,
                    0x73d30bb840406061,
                    0x29814cc4c378b8a9,
                ])),
                Felt::new(BigInteger256([
                    0x55e2ae527f2c0e56,
                    0x481db2b37744949b,
                    0xf13c90d72ef88106,
                    0x1bcf2556e0f1df49,
                ])),
                Felt::new(BigInteger256([
                    0x0c75312e80002dd4,
                    0x92a9199b57db7dd5,
                    0x5760c22804a5b6ff,
                    0x16d0646af98191bf,
                ])),
                Felt::new(BigInteger256([
                    0xe20d38e8a0a866ef,
                    0x8bc381d55f5942c9,
                    0x6bbf25a53cc438fb,
                    0x38135629dd1e6cb9,
                ])),
                Felt::new(BigInteger256([
                    0xdac78ef0b9d22fec,
                    0xcbaa562d06debe7c,
                    0x3074c1d19d4f8824,
                    0x0a8800640804641e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xcfd69bccf236a273,
                    0x4884ffafcad5c044,
                    0x6d8e851dbfcd6091,
                    0x0cf6cc305a81e106,
                ])),
                Felt::new(BigInteger256([
                    0x13e028c2700c2d9a,
                    0x717ef83897fc1191,
                    0xa184d0e2e09bcfb5,
                    0x0adc911462502d11,
                ])),
                Felt::new(BigInteger256([
                    0x9035d6ed0f410824,
                    0xc378ffd90d1e1bf9,
                    0xe88337d69b72e404,
                    0x3265982ac489deb5,
                ])),
                Felt::new(BigInteger256([
                    0x90f07a2cfdb7b7b3,
                    0xc3ec810470bd3c2c,
                    0xeb1d1a3c997f2733,
                    0x0d5ecb7ad960cd17,
                ])),
                Felt::new(BigInteger256([
                    0xe2d2e655540c2132,
                    0x7546a5734eec7491,
                    0xb789e269665abc89,
                    0x152e5936a2e2679c,
                ])),
                Felt::new(BigInteger256([
                    0x1450aee770d79a09,
                    0xd46a6bb9243db296,
                    0x49b30fd45e50eabb,
                    0x35c142726f6dc209,
                ])),
                Felt::new(BigInteger256([
                    0x4b0c5d2c1209b8ea,
                    0xa0fd7546ca06c31d,
                    0x7c528dd5a769c40b,
                    0x20f06c28f1323b69,
                ])),
                Felt::new(BigInteger256([
                    0x19833e83cead58cb,
                    0x4a0b065d1e481920,
                    0xd453f9be74ff05e4,
                    0x3f082717b1eacaa0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc83bb16c2963dff7,
                    0xdc6215de5331eda3,
                    0x6a87b8d94d3b199e,
                    0x29b07e92f8bea2e4,
                ])),
                Felt::new(BigInteger256([
                    0x20e5429707d60362,
                    0x1e64ddda7d63b055,
                    0x242f0a713626cdd5,
                    0x24b093a33ea21b01,
                ])),
                Felt::new(BigInteger256([
                    0x6bdfcd8781ecb928,
                    0xa2a1779eb7d74ed2,
                    0x532e1b7931638efa,
                    0x164a7d3354c3e1d4,
                ])),
                Felt::new(BigInteger256([
                    0x8b5c42bbe7fdd9a2,
                    0x52df0265b16b557b,
                    0x4f6712adc9681eea,
                    0x05749bf56d2155e4,
                ])),
                Felt::new(BigInteger256([
                    0xdf68267c4ae65e0d,
                    0x3652805c98271ef6,
                    0xef8b8e535cf4892d,
                    0x17d3c7b5b85a48e0,
                ])),
                Felt::new(BigInteger256([
                    0xc322a8b2007256cd,
                    0x49e0d9a694175a85,
                    0x2e43830b1e1bc572,
                    0x09f6d3353c89fc70,
                ])),
                Felt::new(BigInteger256([
                    0x60adae6f68e5afc6,
                    0x49e9b05b318d53e6,
                    0xbc11bec74551f400,
                    0x1ce75d7626ac7810,
                ])),
                Felt::new(BigInteger256([
                    0x5674cab521bd7150,
                    0xfdcf2a340cff5cfe,
                    0xaa3e599f14970929,
                    0x135ced83e5d73730,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x286d44c0a9e8ea28,
                    0x12bac5750d274776,
                    0x24b1e0e2ba5e03fe,
                    0x0d9f73f1ff8d8378,
                ])),
                Felt::new(BigInteger256([
                    0x0eb9d327594cb070,
                    0xc6d50cb993ccc926,
                    0x3ae75687d0bc3359,
                    0x2616cf23a6128a8f,
                ])),
                Felt::new(BigInteger256([
                    0x5c86f27861f9bbcf,
                    0xd86e3dc66fc42930,
                    0xf2808745f8174fef,
                    0x0fb3bfa070731dea,
                ])),
                Felt::new(BigInteger256([
                    0x5c6c52e95a08cdac,
                    0x95cfdd89c0c172b5,
                    0x4e1b3a2e4b51b92f,
                    0x1532733a44cfdf4c,
                ])),
                Felt::new(BigInteger256([
                    0x1d5e852cb3634db7,
                    0x66fb249250a79c5b,
                    0xb4a583af3ad8d95a,
                    0x37669359e49ef0e0,
                ])),
                Felt::new(BigInteger256([
                    0x8c59d2a0dbedeab1,
                    0xa224619fa29467df,
                    0xee15ace92a63fc95,
                    0x1d3b4dc571c5374a,
                ])),
                Felt::new(BigInteger256([
                    0x8f8d7865dcb7d573,
                    0x35b315882f11b31a,
                    0x51895d8f6a872d58,
                    0x1cb49db025fd1b43,
                ])),
                Felt::new(BigInteger256([
                    0x105b7ed9c4dcf781,
                    0xa760611fef7dac68,
                    0x6f9373763f41cf71,
                    0x3c3e14565bf81829,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x627fc5eaf39d71c1,
                0x18c6473493dc1f36,
                0xad2b6c19e77d78da,
                0x0fb523453753a52c,
            ]))],
            [Felt::new(BigInteger256([
                0x92240aea65dab062,
                0x99cc0e033a1e2c42,
                0x81c5f3435fd86830,
                0x1e24d0d505c77292,
            ]))],
            [Felt::new(BigInteger256([
                0x960b7c90f4509ab4,
                0xfae7f25ce681694c,
                0xb1ca036e73f04a0e,
                0x3a0969c1410653a5,
            ]))],
            [Felt::new(BigInteger256([
                0x488a46c7e10f0110,
                0x3d3546809c894326,
                0x9216737073d0a439,
                0x3fa2bb5e69f9c0cc,
            ]))],
            [Felt::new(BigInteger256([
                0xb47df19a86675f04,
                0xb474541fb0e942d7,
                0xc0878e45f39d61b2,
                0x1d3520cc25aca36b,
            ]))],
            [Felt::new(BigInteger256([
                0xb76c81890eb065e0,
                0xd4d5916166bed151,
                0xcb948723c1d54b70,
                0x14c1ae15efde6ec8,
            ]))],
            [Felt::new(BigInteger256([
                0x5a170975a53ed7ab,
                0xf16d247e7a442708,
                0x2178def1683f854c,
                0x1e46c8bd3a4627d1,
            ]))],
            [Felt::new(BigInteger256([
                0x7832c1a93de38176,
                0x9ded7d0c689e81d2,
                0x6719ba80149c3e7a,
                0x278abba0e9828fe0,
            ]))],
            [Felt::new(BigInteger256([
                0xe826e5cb80cf6015,
                0xf9378e53d283de3c,
                0x14a44b52973cb764,
                0x1fb3dc0fc8f82ef3,
            ]))],
            [Felt::new(BigInteger256([
                0x3ddfdc1f0e68ec29,
                0x3feabb6626236f88,
                0x81380937a6d00843,
                0x3fef5a134b9be703,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 8));
        }
    }
}
