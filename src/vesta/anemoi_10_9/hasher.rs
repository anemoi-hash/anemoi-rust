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
            vec![Felt::zero(); 10],
            vec![Felt::one(); 10],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
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
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger256([
                0x3b63640f847d65ce,
                0x40e7dd67f1e67c16,
                0x9e35f2e32f086203,
                0x20de472d91e983d4,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0xb26af301881e8578,
                    0xf1b2e16fb2ac4e66,
                    0xb44b65fdba02e059,
                    0x26f9c7ed47b30122,
                ])),
                Felt::new(BigInteger256([
                    0x85ac91da6dacb4cc,
                    0xbfe68b96e6e5fd69,
                    0x7c824c4dc07f90c3,
                    0x1231e7c8781628c1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0f75c584a3a88e6e,
                    0x3099c972e96946e9,
                    0x4be37975ab630c05,
                    0x0f7e82e608049b9e,
                ])),
                Felt::new(BigInteger256([
                    0xe5f3e917b9e3152c,
                    0x125d1ac1c040275d,
                    0x096da3280cd48b6e,
                    0x2afe656c45cb5eca,
                ])),
                Felt::new(BigInteger256([
                    0x8759783f1d7df6af,
                    0xe12d1a250df3baca,
                    0x67af34338260af69,
                    0x2b32dcff0b52d3f5,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7615a1d984863779,
                    0x1b98564a60651830,
                    0xf3990988caa7cbcb,
                    0x1082242dc699e82a,
                ])),
                Felt::new(BigInteger256([
                    0xc0a3ef62bca8e7cd,
                    0xd6ff52c078fb9e44,
                    0xfa814a78f22fb77d,
                    0x39f899bcaf290d63,
                ])),
                Felt::new(BigInteger256([
                    0xa60fb4a1deb4006f,
                    0x2451cc39f880fed6,
                    0x47e4a2e0e8aee60e,
                    0x01e2305ebb62cff4,
                ])),
                Felt::new(BigInteger256([
                    0xd7264e58a3a8a9f0,
                    0x152310dad9094e3a,
                    0x94640e0f7d41f4f1,
                    0x087e67bdab350a67,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xeed834bc3c5cee6b,
                    0x267e0241b406aa96,
                    0x904731ba495b59ea,
                    0x1cb350d66073f77f,
                ])),
                Felt::new(BigInteger256([
                    0xf43db48b943ea5b1,
                    0x99bf30315398e020,
                    0x59622b874a461d1c,
                    0x123fe36a2c58e2d8,
                ])),
                Felt::new(BigInteger256([
                    0xc7e5f95d295d2747,
                    0x6e5b8163188e5a29,
                    0xd8e5a10f40819a52,
                    0x1d70f2ad2304db06,
                ])),
                Felt::new(BigInteger256([
                    0x276c1b954bbf21c8,
                    0xd3cda578c69c9509,
                    0x7c344a72b3ed154f,
                    0x3132adf398d4ae5f,
                ])),
                Felt::new(BigInteger256([
                    0xb2e94535eb126c83,
                    0x025828af14db7a23,
                    0x7d1ba907b55a906c,
                    0x15de216a69b44e35,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd92c1363bc11ef3f,
                    0x672dafe39dbfc038,
                    0xd5949e4b65d6820c,
                    0x018291eda9a89cce,
                ])),
                Felt::new(BigInteger256([
                    0x26134f8fbca2b4d9,
                    0x5e7fb68e49629bb7,
                    0x3e5d6f05235efd7d,
                    0x30168e66500f5ba6,
                ])),
                Felt::new(BigInteger256([
                    0x7fd9cac77a67b1d0,
                    0x50571cdf10632df8,
                    0x2313e5eed53c54e0,
                    0x220d9c2820e7c9b3,
                ])),
                Felt::new(BigInteger256([
                    0x8c1e5c23b02ef49a,
                    0x02a37173d992733b,
                    0xaa1da3d8b648c4e4,
                    0x219930e102d83b91,
                ])),
                Felt::new(BigInteger256([
                    0xd9770cc85f68bf68,
                    0x61f7b6ae3a6fbaf2,
                    0xb324d148e06355ec,
                    0x3ae0c5638f191c80,
                ])),
                Felt::new(BigInteger256([
                    0x2222c4434e47c4ff,
                    0x1bed2357407066dd,
                    0xc1abd62a558bdb2c,
                    0x0864a04fa5014efa,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x778e22e8d87fa181,
                0x2940a740595b34f1,
                0x231f797d57131ebb,
                0x32a0b1089c7c9704,
            ]))],
            [Felt::new(BigInteger256([
                0xe425ab826c901f3a,
                0x86bb79ab3acfe8ce,
                0x9a22db6567191114,
                0x301008da0376cdf2,
            ]))],
            [Felt::new(BigInteger256([
                0xdd2a1b8e1a6930c3,
                0xfa9d1ea17f6f7da1,
                0x0d93fef98091e111,
                0x0b423eb25d06c1bf,
            ]))],
            [Felt::new(BigInteger256([
                0x4d10480a2f5d0d47,
                0xf924b96a350cdfd2,
                0x97d9bf44c672f503,
                0x3e17d103085277fe,
            ]))],
            [Felt::new(BigInteger256([
                0x1f02888b583f28e8,
                0x656daa8bdbfa34ec,
                0x162fb438ba574691,
                0x30da662e16cc51eb,
            ]))],
            [Felt::new(BigInteger256([
                0xbe342222b6f7d1e5,
                0xc0a14d6dad293170,
                0xf7f4e3febda4d167,
                0x116194a46e6d2aeb,
            ]))],
            [Felt::new(BigInteger256([
                0xc0d03fa9f13f479c,
                0x8bd03238b3a68693,
                0x70ec22e465f8c948,
                0x3d5f3d67e5f9be8d,
            ]))],
            [Felt::new(BigInteger256([
                0x1a4e7efcaddda462,
                0x02f5fd3de354e513,
                0xca4c245ead7f6681,
                0x112457155e5ed063,
            ]))],
            [Felt::new(BigInteger256([
                0x970d626d6c6ae75c,
                0x7bbcea06b871915f,
                0xcfdb8f6389d1162a,
                0x1b0174c7ada265cb,
            ]))],
            [Felt::new(BigInteger256([
                0x805da4ac8a7e6912,
                0xd3b261ef44ee051b,
                0xab8fdcb13a0b255a,
                0x0bac4412bc40a25f,
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
            vec![Felt::zero(); 10],
            vec![Felt::one(); 10],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
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
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x778e22e8d87fa181,
                0x2940a740595b34f1,
                0x231f797d57131ebb,
                0x32a0b1089c7c9704,
            ]))],
            [Felt::new(BigInteger256([
                0xe425ab826c901f3a,
                0x86bb79ab3acfe8ce,
                0x9a22db6567191114,
                0x301008da0376cdf2,
            ]))],
            [Felt::new(BigInteger256([
                0xdd2a1b8e1a6930c3,
                0xfa9d1ea17f6f7da1,
                0x0d93fef98091e111,
                0x0b423eb25d06c1bf,
            ]))],
            [Felt::new(BigInteger256([
                0x4d10480a2f5d0d47,
                0xf924b96a350cdfd2,
                0x97d9bf44c672f503,
                0x3e17d103085277fe,
            ]))],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 310];
            bytes[0..31].copy_from_slice(&to_bytes!(input[0]).unwrap()[0..31]);
            bytes[31..62].copy_from_slice(&to_bytes!(input[1]).unwrap()[0..31]);
            bytes[62..93].copy_from_slice(&to_bytes!(input[2]).unwrap()[0..31]);
            bytes[93..124].copy_from_slice(&to_bytes!(input[3]).unwrap()[0..31]);
            bytes[124..155].copy_from_slice(&to_bytes!(input[4]).unwrap()[0..31]);
            bytes[155..186].copy_from_slice(&to_bytes!(input[5]).unwrap()[0..31]);
            bytes[186..217].copy_from_slice(&to_bytes!(input[6]).unwrap()[0..31]);
            bytes[217..248].copy_from_slice(&to_bytes!(input[7]).unwrap()[0..31]);
            bytes[248..279].copy_from_slice(&to_bytes!(input[8]).unwrap()[0..31]);
            bytes[279..310].copy_from_slice(&to_bytes!(input[9]).unwrap()[0..31]);

            assert_eq!(expected, AnemoiHash::hash(&bytes).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 10],
            vec![Felt::one(); 10],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
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
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x24325c98d927986a,
                    0xd74e1fc2b0805586,
                    0x1c1bc6195139d352,
                    0x35439f9979826fca,
                ])),
                Felt::new(BigInteger256([
                    0x6105c9ad86b0e5a8,
                    0xdd1a9b70fd313c78,
                    0x3eb9382dfe22edd1,
                    0x39af16d03f0e3ac7,
                ])),
                Felt::new(BigInteger256([
                    0x3a5c4225983d3620,
                    0x6b83641545b76b98,
                    0x683099f3cf905827,
                    0x295ff3934d65cb51,
                ])),
                Felt::new(BigInteger256([
                    0x1be6f718df849155,
                    0x3922b209a8779bb5,
                    0x272ad2f63fb35af8,
                    0x0c5920a593a3f881,
                ])),
                Felt::new(BigInteger256([
                    0x5e2f5ca92701ba14,
                    0x49be465e5d6ac13e,
                    0x13f66a32c2c8036d,
                    0x0d02dfdfa47212c8,
                ])),
                Felt::new(BigInteger256([
                    0x3ca32614c9505572,
                    0x9b009cb792186927,
                    0x301de5a9a26837c1,
                    0x08c8b73aae5581d8,
                ])),
                Felt::new(BigInteger256([
                    0x8902d5acb9b84026,
                    0x65cbc7c5a6f3fee9,
                    0x0a1c5f3f75d437b9,
                    0x3a1a2b8db5a51761,
                ])),
                Felt::new(BigInteger256([
                    0x4c92113ff575d409,
                    0xe5a33ba67c5922b8,
                    0x2686b978d3ba3225,
                    0x308da064223e2d67,
                ])),
                Felt::new(BigInteger256([
                    0x3098081e059b4c45,
                    0x79ecc3843ae236b2,
                    0x8239ea688290ea04,
                    0x1fb2ca3943810e6d,
                ])),
                Felt::new(BigInteger256([
                    0xdf0e0e09b45ca4a3,
                    0x7fdc63fcf767b40f,
                    0x36009ce1555e8cbe,
                    0x3d748d619c610040,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2734367812c2b727,
                    0x14500b71646b9ead,
                    0x1d778a2ba7ecee22,
                    0x06b009888fc98ae5,
                ])),
                Felt::new(BigInteger256([
                    0xa3e57777a7b31844,
                    0x62a899cd0c728414,
                    0xf6316e7e36cacb36,
                    0x214720809369091a,
                ])),
                Felt::new(BigInteger256([
                    0xf60c3d5e8896fda0,
                    0xb63c44e2f0b0bcd3,
                    0x0bfe65049c6fe64a,
                    0x3a0e2f98a619c14b,
                ])),
                Felt::new(BigInteger256([
                    0x663a82ea67dc0e38,
                    0xbc4c43d6da9de34c,
                    0x9a79f835f8037800,
                    0x2da1bae6611a08a0,
                ])),
                Felt::new(BigInteger256([
                    0x5cd69e8fb7194d58,
                    0x6c0f8513ded41b57,
                    0x66bb8d68231326bf,
                    0x3e21c097b55621aa,
                ])),
                Felt::new(BigInteger256([
                    0xc5266955da3eb7c9,
                    0x6e6e48541db7d5b1,
                    0xc68b6667fdbbb071,
                    0x26fb2137166e26d5,
                ])),
                Felt::new(BigInteger256([
                    0x0f9816fbd85d4d53,
                    0x7881411a404210df,
                    0x61e1a5ca5cf8c88d,
                    0x0fe073ff4ff19bdc,
                ])),
                Felt::new(BigInteger256([
                    0x53154952aefadb60,
                    0x84026b7afd2d8633,
                    0xeb8e35ca989317da,
                    0x399627f497359690,
                ])),
                Felt::new(BigInteger256([
                    0x266bb6e294d68290,
                    0x7909f04fd3555186,
                    0x1bee6605734bfb37,
                    0x0a268be49bbb3853,
                ])),
                Felt::new(BigInteger256([
                    0xe8e3b0611b9bb81f,
                    0x5f6b22f8a7e8faf9,
                    0xee0650063d783d72,
                    0x32bf44d31bde44e6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xdb7f3cbe1a6183a5,
                    0xbd57058e0bb1bfd9,
                    0x90c28a37877242fa,
                    0x310cb787ceb0fe15,
                ])),
                Felt::new(BigInteger256([
                    0x2bf3075b0b134e06,
                    0xfce7d725e32ee7fa,
                    0xc401cab4ada7626a,
                    0x234d568211880773,
                ])),
                Felt::new(BigInteger256([
                    0x646c2b1c29da51e2,
                    0xaa3a1f95ae571d39,
                    0x5e0ad693da547378,
                    0x39a451e57f2f521f,
                ])),
                Felt::new(BigInteger256([
                    0x1789db47c0f39b8b,
                    0x007c5a99aed70125,
                    0xe64213156ed3cd64,
                    0x17bc2df2e7255024,
                ])),
                Felt::new(BigInteger256([
                    0xe59c5b57658580b9,
                    0x540677f062fa8b19,
                    0xf31b522a784f79e9,
                    0x1f1e67fbbdb774ca,
                ])),
                Felt::new(BigInteger256([
                    0xdf7cd326cfec3b35,
                    0x0b7789e626c8c770,
                    0x0f82ff3072003cd4,
                    0x0469190453ddd7a7,
                ])),
                Felt::new(BigInteger256([
                    0x66ee389e0d5285f1,
                    0xd13e6a4adb8e6a85,
                    0x346ae89ab80850af,
                    0x3d6de880f67eb63b,
                ])),
                Felt::new(BigInteger256([
                    0x0e85a1f4f55554e1,
                    0x62c2ebd4771defd9,
                    0x62ba62554ecda0c4,
                    0x175b400f0eddb0e1,
                ])),
                Felt::new(BigInteger256([
                    0xc25770e6ca31db0b,
                    0x1534af053c6bda04,
                    0x64e597d805ef4290,
                    0x34fc18c9e9688e29,
                ])),
                Felt::new(BigInteger256([
                    0xa9db0e0be9488833,
                    0x850a336cc92441a4,
                    0x191dce4d044ae80b,
                    0x0b876c9318cfcfea,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x33ca234c2ca522b8,
                    0x4ff661892c5cced5,
                    0x852f94b0f96d1187,
                    0x3a74e4287a45a856,
                ])),
                Felt::new(BigInteger256([
                    0x04e9edf60471e2b4,
                    0x5018fc0448c9a6a6,
                    0x4255ea4176d189b6,
                    0x38b18a786cec000d,
                ])),
                Felt::new(BigInteger256([
                    0xa55d84a78d10e983,
                    0x44436dd5fcf6b861,
                    0x5bd036333ae87ff9,
                    0x1eead4b1934943c9,
                ])),
                Felt::new(BigInteger256([
                    0x006e5e091978fca0,
                    0x4443b86f6a42870f,
                    0xd2c5aa0e37bc10a4,
                    0x23849f7ab4a76520,
                ])),
                Felt::new(BigInteger256([
                    0xa270bb9a2f547681,
                    0x474fb421c6fe411e,
                    0x3031d3f1b8efc676,
                    0x07fdeb66c23f7646,
                ])),
                Felt::new(BigInteger256([
                    0xaae6af6b965b8132,
                    0x4b8bfd77c2de9e4b,
                    0x4c0d36f6e0d283e3,
                    0x223f2252b38beb46,
                ])),
                Felt::new(BigInteger256([
                    0x6dbbefeafde3c047,
                    0x1da679b78e47b6c8,
                    0x01fd8b92ba1ab206,
                    0x1d6c41d959b66ba5,
                ])),
                Felt::new(BigInteger256([
                    0xd4d43a20b20f6b98,
                    0xa3ce01725bbd8d95,
                    0x92aeea05dbad7acb,
                    0x17cb95ba5b97706d,
                ])),
                Felt::new(BigInteger256([
                    0xdcc73684b3a2fd3a,
                    0x27021a22dbe27a52,
                    0x64f5d1f23f69ec47,
                    0x004800b24acb9c84,
                ])),
                Felt::new(BigInteger256([
                    0x65f26ad7dd9deaa1,
                    0xa69b5858ef0d8ead,
                    0x23afaf9370d9b601,
                    0x24d3583ba85e603b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x83f6fa03ab8fbb08,
                    0x54469d2a534bf109,
                    0x37a61ce99c234757,
                    0x21b467f5331df4bc,
                ])),
                Felt::new(BigInteger256([
                    0x69f2b1a9577cc751,
                    0x788156994c4f9eb8,
                    0xadad4393820e285f,
                    0x0b190cd2ac68340b,
                ])),
                Felt::new(BigInteger256([
                    0xf2c8cc962235c3e7,
                    0x401a8e757ff8aa0a,
                    0xf1c4870f4aa2faf2,
                    0x2d2631641865ac1a,
                ])),
                Felt::new(BigInteger256([
                    0x5a4608584a9528bd,
                    0x547e2c9695ee3602,
                    0xaeba705761d8e755,
                    0x088ac5668a4611b0,
                ])),
                Felt::new(BigInteger256([
                    0x92744d27c83de2fb,
                    0xb11e4cecbb0a9de2,
                    0xf439bebbb4978e0e,
                    0x3615f817969d70ad,
                ])),
                Felt::new(BigInteger256([
                    0x47694f458dcc764a,
                    0x02e55c4f098f3bb3,
                    0xe082a93154d924e9,
                    0x0c125cb290bee912,
                ])),
                Felt::new(BigInteger256([
                    0x004a282f25014598,
                    0x51b6b64c988a3614,
                    0x3bc1c4961ee43baf,
                    0x2863b41275fefd64,
                ])),
                Felt::new(BigInteger256([
                    0xcf161a5055bf425c,
                    0xda4838a711336859,
                    0xab587927a2575688,
                    0x274edffe96968862,
                ])),
                Felt::new(BigInteger256([
                    0x93580ac842509a95,
                    0xf49e3060eefba673,
                    0xee908a55972199b0,
                    0x2781c397a69ca33c,
                ])),
                Felt::new(BigInteger256([
                    0xf029eb616e65ce89,
                    0xcee1c2787bc31ba9,
                    0x60ab8c98fa2267cb,
                    0x1b778f921cdf486a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5ad97142e13c7515,
                    0xf943dcc5ee22c716,
                    0x980163c298086ea1,
                    0x2a3caed79b852607,
                ])),
                Felt::new(BigInteger256([
                    0xdaba0179305c46e6,
                    0xf0522b4eeb872d58,
                    0xddee7277d641ed10,
                    0x228092c36931c410,
                ])),
                Felt::new(BigInteger256([
                    0x2ce06857016d76e4,
                    0x804ff3cddb1ad709,
                    0xf70cae75bf4f53cb,
                    0x17c3aa332f030737,
                ])),
                Felt::new(BigInteger256([
                    0x3fee4440c6854e57,
                    0xfccf653de5534add,
                    0xad636b481778615a,
                    0x3b131d08c158d317,
                ])),
                Felt::new(BigInteger256([
                    0x6f7ba3b70424c94e,
                    0x912d1b4068727a2a,
                    0x1352a6dba66953c5,
                    0x36001f7be3626843,
                ])),
                Felt::new(BigInteger256([
                    0x136be91ed4d420db,
                    0x478e090a2d1ad962,
                    0xa5010d7ae82e29df,
                    0x1554589eb648871c,
                ])),
                Felt::new(BigInteger256([
                    0x4ed1a13c08faa50b,
                    0xc140873ffe88611f,
                    0x81adc7733f34f21e,
                    0x03d72f8f77938535,
                ])),
                Felt::new(BigInteger256([
                    0xdc65ef3c591a5191,
                    0x4f17d8f9ea87d02c,
                    0x5b991021561e1fe2,
                    0x254638623e561f5f,
                ])),
                Felt::new(BigInteger256([
                    0xea260e808b26460b,
                    0x8dd6f36134ee62ad,
                    0xede47697408015c9,
                    0x0fa337e5c47910b5,
                ])),
                Felt::new(BigInteger256([
                    0xe63a73afa7d41a79,
                    0x0303bf3c5019c4f8,
                    0xdbddb72451ee8038,
                    0x2f9a57a6c24fb54e,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xeb6cef1b24fa3323,
                    0x0dc6d0ed41fb9c42,
                    0xa75f3318b1df198e,
                    0x2394327b32a4164a,
                ])),
                Felt::new(BigInteger256([
                    0xd9e2b2bdc1c2a877,
                    0x4345d39260474821,
                    0x5e80e4ba7e886156,
                    0x2ef9c5e75fc0d87d,
                ])),
                Felt::new(BigInteger256([
                    0xdabd7a68c0db1bfb,
                    0xd78d6f0a67e4e97a,
                    0xbc71b6f1cd0437ac,
                    0x2a5971f44671ce5e,
                ])),
                Felt::new(BigInteger256([
                    0x9bfb7705da96283f,
                    0x45a5bd50a8b4b3ca,
                    0x7e527b9f5902e41e,
                    0x02757fbc6bff6911,
                ])),
                Felt::new(BigInteger256([
                    0x3604ace18594c4b1,
                    0x9184a51dc41dd11e,
                    0xa960a06622377337,
                    0x08ed11ecb30e5a17,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd4b1c488b3a78b33,
                    0x596fe3ec75941460,
                    0x26845197e4ce6cc4,
                    0x0d0c509637683329,
                ])),
                Felt::new(BigInteger256([
                    0x66e2e94f2208f122,
                    0x92c7ed03972d9c53,
                    0xce0debd121ddcd1c,
                    0x350c27cd30b2896d,
                ])),
                Felt::new(BigInteger256([
                    0x73225bb4df592449,
                    0x049b5f0226ab46e8,
                    0x7184238de1cd872d,
                    0x2a82e454d0b84465,
                ])),
                Felt::new(BigInteger256([
                    0x75e68d23c690d11e,
                    0x13387f99ae60f11c,
                    0x0ddf29f7990fc9cb,
                    0x26f0928444bb15f1,
                ])),
                Felt::new(BigInteger256([
                    0x9a7796ab1233fb35,
                    0x68b1194e3ba80dcd,
                    0x752e46974eeb4065,
                    0x10b932f56453670d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd93f8f56e06a8c6a,
                    0x6b56d10fcae2b7cd,
                    0x3ae96a6ea2c8d4e6,
                    0x190d07ba6691dcf0,
                ])),
                Felt::new(BigInteger256([
                    0xabfe79fd4c861128,
                    0xb227228ce6aa4dd8,
                    0x08d14bd01ff45bc3,
                    0x0657751fe12a2ca0,
                ])),
                Felt::new(BigInteger256([
                    0x460a6e9131269f02,
                    0x8a5f346dc33d0a86,
                    0xeb9d03b48b6ec8e2,
                    0x0f0e6c7609dd81df,
                ])),
                Felt::new(BigInteger256([
                    0xe437c593ada49289,
                    0xd86b893bbf77886e,
                    0xdc0f9469b7d1040a,
                    0x2927643f8b9259df,
                ])),
                Felt::new(BigInteger256([
                    0xee11ee35119d0d0f,
                    0xf93e457cd17bfe1e,
                    0x0c4c095d31d0f97c,
                    0x0b4dd5b1838ee081,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb4d29bd8ef421d30,
                    0x10462be5a12857a8,
                    0x95a41b6273e9b38a,
                    0x07ebedb1910c2cf5,
                ])),
                Felt::new(BigInteger256([
                    0x7fab60284b5c1ee1,
                    0x6164145bf7c2c2c6,
                    0x7b9fe9bbca29f73e,
                    0x250a16f02f1aecad,
                ])),
                Felt::new(BigInteger256([
                    0xd62426afe9bcff1e,
                    0xa84c6a629b44b3c8,
                    0x9ac37d3a74be71a7,
                    0x266789f11603e65e,
                ])),
                Felt::new(BigInteger256([
                    0x2ff3762011f62f22,
                    0xe7d73d37bb469b9c,
                    0x3d11e1b7adbd153f,
                    0x07e66d7d275653ac,
                ])),
                Felt::new(BigInteger256([
                    0xd300684344ebe6b2,
                    0x7d1db30d9f268c5c,
                    0xbb03674727732562,
                    0x0174a353c2a58a4b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x680f45562c17a367,
                    0xc8431babac171ef6,
                    0x442bb0f786a35ab9,
                    0x1642589082f7134c,
                ])),
                Felt::new(BigInteger256([
                    0xa5242b1695505ffd,
                    0x4479dd6c38922aae,
                    0xaadfd010c47917c6,
                    0x01bcece5619fb007,
                ])),
                Felt::new(BigInteger256([
                    0x0d47d08095625857,
                    0xcaa46aa73ef932b1,
                    0x5023b48a66a3dbaa,
                    0x1f539957889f9eb7,
                ])),
                Felt::new(BigInteger256([
                    0xf3ce39601ccba5b8,
                    0xed31459461c2c0b5,
                    0x918610758f9a054f,
                    0x0ddbf2db25fbcee1,
                ])),
                Felt::new(BigInteger256([
                    0xb24af15bb8b9a68a,
                    0x223bb780807b0472,
                    0x15adb3a68c17c487,
                    0x1641e4d6405c199a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8d273ae53ca48a39,
                    0xadc57b89f2638ba9,
                    0xfd7a4fbf718980b2,
                    0x3a3b71d8aeddb0c1,
                ])),
                Felt::new(BigInteger256([
                    0xbe2cbbc380b25fdc,
                    0xf80d919c9e9ac2c0,
                    0x173d8c076ce6aa25,
                    0x0c43171e071380b9,
                ])),
                Felt::new(BigInteger256([
                    0x21601bc199411941,
                    0x0e4cb2feedf3f310,
                    0x4a657ab46bc1a0a3,
                    0x165f3d1de2de592e,
                ])),
                Felt::new(BigInteger256([
                    0x14614726fdf6cd1d,
                    0x47ec48edc8ea4aff,
                    0xc8be110f6fe887ef,
                    0x077fe042d730bfdb,
                ])),
                Felt::new(BigInteger256([
                    0x631a205803863d39,
                    0xd1f33ade6eae07ce,
                    0x183a5a31c6525093,
                    0x37d0a8fd953e6b96,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbfb41364978d0ec6,
                    0xcf7b97b4d960b808,
                    0xb24a562a69842738,
                    0x2538ae8b88afc40e,
                ])),
                Felt::new(BigInteger256([
                    0xf3eeca3587ad0144,
                    0x3d4104047a7a7a7a,
                    0xb1f79cbe119f1b68,
                    0x0d58da12703a237d,
                ])),
                Felt::new(BigInteger256([
                    0x581ae61ab570bee1,
                    0xe9793d46e8c73c96,
                    0xb577f7e5ad066619,
                    0x23b1c8f13d36a4fd,
                ])),
                Felt::new(BigInteger256([
                    0x180f41791d815fde,
                    0xa562253b7dcd0661,
                    0x92704da3c669ecc9,
                    0x11a42259fbcb464c,
                ])),
                Felt::new(BigInteger256([
                    0xe02dceee07c449fd,
                    0xdd3b667c64604314,
                    0x1731054543c0ccc5,
                    0x15a6fe60d19a51d2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x655c0a1bdaf0dd86,
                    0x1598a42f8c5408e9,
                    0xfeb6f06c555cbadd,
                    0x2e9c5776588b5ac4,
                ])),
                Felt::new(BigInteger256([
                    0x409fc29cecb06621,
                    0xdec6f78be82e9de7,
                    0xa8998d23bf8841a8,
                    0x349c09b6b785ef7d,
                ])),
                Felt::new(BigInteger256([
                    0xc11e2540bd78a4f2,
                    0xdf20a04bf29f38b7,
                    0xb642570a2505e065,
                    0x3a7a9baca29178c7,
                ])),
                Felt::new(BigInteger256([
                    0x4894063f164f78a7,
                    0x11470dd8bb6873d6,
                    0x8abd5519016fd57e,
                    0x3af667622d3d0d5f,
                ])),
                Felt::new(BigInteger256([
                    0xac80757ff02d72a3,
                    0xb26d639fff204978,
                    0xf4c4d10215bc1ef2,
                    0x073643b9ecd7997a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1b46b48846279beb,
                    0x5b09a3d1438f58a3,
                    0xa0b0ea1edb1f3ee9,
                    0x02f8a54406e56984,
                ])),
                Felt::new(BigInteger256([
                    0x1bb0f0d3ccda7d80,
                    0x33bbfd67ffd1e79f,
                    0x00beb6fd1e49f9ae,
                    0x3f13f839d4756524,
                ])),
                Felt::new(BigInteger256([
                    0x4e2572f70cbb404d,
                    0x67176d821fb5d93d,
                    0xfb4a48716dbec18e,
                    0x3a64f2a90f2c87d1,
                ])),
                Felt::new(BigInteger256([
                    0xd627452f68adc305,
                    0x4f2160393ce4f122,
                    0x5871c8cb221cd161,
                    0x0eb24179d062ebaa,
                ])),
                Felt::new(BigInteger256([
                    0x0b248c4047b1785b,
                    0x6c66762874ac761d,
                    0xd7c40ae7f537b027,
                    0x0752e6c694dcb836,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7754429448ee2690,
                    0x5e238323de84aa48,
                    0xa4a0e36e5b943199,
                    0x2b985b96fd7e4bad,
                ])),
                Felt::new(BigInteger256([
                    0x2feec0bf88613191,
                    0xbdacb66b462fa6ad,
                    0x88fdc929642be438,
                    0x1644ab42aefde1e2,
                ])),
                Felt::new(BigInteger256([
                    0x92fed4480bf45bea,
                    0x826f95abb0b38e89,
                    0xccc80495260a4585,
                    0x3a7a80aa4da5da7f,
                ])),
                Felt::new(BigInteger256([
                    0x7159e35cc2e36f23,
                    0xc9cb2e15802db907,
                    0xbfb2252a5ca1ea91,
                    0x096c8812b008a294,
                ])),
                Felt::new(BigInteger256([
                    0x267d45b30b432454,
                    0xeeaa06670110b90d,
                    0xc2300239cc5504df,
                    0x1dadc979e28f9784,
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
            vec![Felt::zero(); 10],
            vec![Felt::one(); 10],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
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
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x65976af4697a4d6f,
                    0xee94a8a67fee9df6,
                    0x003ffd1b455ef3e7,
                    0x23f211ec2e313e6e,
                ])),
                Felt::new(BigInteger256([
                    0x94e24b9c08acf449,
                    0xc344849876afa7fc,
                    0x94ff4fcbd1e83a93,
                    0x066fea96deab0478,
                ])),
                Felt::new(BigInteger256([
                    0xd4a8c8a3eaa54ca3,
                    0xca3aae7cfb58fb2d,
                    0x96c3214ce5521e2d,
                    0x0816be8a4b49827b,
                ])),
                Felt::new(BigInteger256([
                    0x416ea3a0884273ab,
                    0xe59e5415fe323f64,
                    0x6d07cd3aa747ff6b,
                    0x3a0a9211893c2a40,
                ])),
                Felt::new(BigInteger256([
                    0x44d4777d8e9a1b59,
                    0x8c8f3cd714c31c6d,
                    0xd75f8119de5d8aac,
                    0x1061aaf69982f3d4,
                ])),
                Felt::new(BigInteger256([
                    0xd80da9366f75a97c,
                    0x22092dc15b741103,
                    0xeb74197f71c86294,
                    0x2caea3d13d1c9def,
                ])),
                Felt::new(BigInteger256([
                    0x75ab056c1bf08b77,
                    0x72396484c35c51fd,
                    0x8b6991e574080da2,
                    0x2ec7cc00b8880de3,
                ])),
                Felt::new(BigInteger256([
                    0x706aeccfba4973ab,
                    0x0f6a3985a60b0925,
                    0x768c1ce5cf875e8e,
                    0x39de4d05ca8f8c6c,
                ])),
                Felt::new(BigInteger256([
                    0xe6f7788d479b382a,
                    0xb430c1faa5d50d8e,
                    0x7e12a726c576b6b2,
                    0x2739b18008c341e9,
                ])),
                Felt::new(BigInteger256([
                    0x23200c0f9923a5d2,
                    0x67d059cb470f4043,
                    0x359fe53e8da5d4b7,
                    0x353f26d021e99924,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x90c267eabe138e63,
                    0x6ef6572501b0a527,
                    0xd2e31527d4befd79,
                    0x16312cbd4b3a9f33,
                ])),
                Felt::new(BigInteger256([
                    0xc7a9d4364525a100,
                    0x73cc3e8712f44efc,
                    0x35f87f53f11a47fa,
                    0x1cc2f87063c71ea5,
                ])),
                Felt::new(BigInteger256([
                    0x5d87a190ee927b5c,
                    0x6510a98f3a5d4eb5,
                    0xa455803bf3ca5eca,
                    0x18fcfa1551e9ecc8,
                ])),
                Felt::new(BigInteger256([
                    0x7a99112f737f393f,
                    0xc8e2b50db2587df6,
                    0x756635f362d36357,
                    0x060d0589496a2ea0,
                ])),
                Felt::new(BigInteger256([
                    0x05f9f34eb803d55e,
                    0xcfbf77bbf67eaed0,
                    0x3ace2bba7552cdae,
                    0x1a9be7d6bad85a55,
                ])),
                Felt::new(BigInteger256([
                    0x7e7cb89ccd0d7802,
                    0x41d7189ecc80c429,
                    0x696648a12590b17f,
                    0x1d6eb10db2b9df31,
                ])),
                Felt::new(BigInteger256([
                    0x89615a69e997d901,
                    0x5f208a0c5602fc3a,
                    0x7d225857fa38744d,
                    0x18021823593476f2,
                ])),
                Felt::new(BigInteger256([
                    0x9ee2b1be9d076b77,
                    0x1dc76358759c9081,
                    0x4be5be0f1f1dce1c,
                    0x12e89d20e6b003c0,
                ])),
                Felt::new(BigInteger256([
                    0x3f7645bff4268ee9,
                    0x7f15f053961e335c,
                    0x97f186cff92028e7,
                    0x1ff63f63ec03850a,
                ])),
                Felt::new(BigInteger256([
                    0xa25850f56edaf60e,
                    0x1d96fb99d5a20645,
                    0xa80ed2eddbb0c97e,
                    0x1d3a172e4c00388b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5130479e49c065d1,
                    0x0c946d1df7c95676,
                    0x812d8717dfe292d1,
                    0x3b290258bb4309d4,
                ])),
                Felt::new(BigInteger256([
                    0xcae746f64b70df87,
                    0xa9334ff10d01e2a0,
                    0x5c8dd91c6263d03f,
                    0x262e135861455696,
                ])),
                Felt::new(BigInteger256([
                    0xa0f14b27493b1591,
                    0x295ffbdc4f370717,
                    0x037134cf7acefac7,
                    0x2860b22ccec6127a,
                ])),
                Felt::new(BigInteger256([
                    0x02334bee8450cafa,
                    0xaf3f2242f6369f8a,
                    0x2386573606de3b07,
                    0x3defade02eb3e46e,
                ])),
                Felt::new(BigInteger256([
                    0x3dc038f773485e86,
                    0xdb693371e8aa3860,
                    0xa2c7781f4d12b897,
                    0x136dfd1cbd323712,
                ])),
                Felt::new(BigInteger256([
                    0xa1caa4a81120618f,
                    0xa7cd0fe125d78da1,
                    0x26e2a929c19cead1,
                    0x2a16059cf090fe4f,
                ])),
                Felt::new(BigInteger256([
                    0x48cf0c2771cc888f,
                    0xa106ef8c6ad3ab9f,
                    0xbc45454903051962,
                    0x06127493ff503d58,
                ])),
                Felt::new(BigInteger256([
                    0xf305b1a472e12924,
                    0x59793bba21e7d4c7,
                    0x755d9201f06b2a12,
                    0x3818b2ade0f2097a,
                ])),
                Felt::new(BigInteger256([
                    0x6c604c0ba1609205,
                    0x840ef04ce275cde7,
                    0x53d433695638349a,
                    0x3f4497adf0dab035,
                ])),
                Felt::new(BigInteger256([
                    0x78c6018247a1ab3f,
                    0x4a4fdcb1b780c0d2,
                    0x1e0c61e21b92a7fa,
                    0x21b786880aea2505,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xfc6abcbd129bf759,
                    0x9a333106876e2a39,
                    0x88f4823dc4165381,
                    0x2ae01fccb8949e94,
                ])),
                Felt::new(BigInteger256([
                    0xc2407e09fac62ffa,
                    0x5b25d23b6af66d8b,
                    0xb2fb4416aaf08330,
                    0x25ab2cbed6301bdc,
                ])),
                Felt::new(BigInteger256([
                    0x5ae1e07cf55c882b,
                    0x073ba676ecce187a,
                    0xf26bdd5586b0e4ba,
                    0x3ddad43746b8c9f2,
                ])),
                Felt::new(BigInteger256([
                    0xde49bf45f95a0e1b,
                    0x51d3ee4c62f35399,
                    0xedb87ad1166130c6,
                    0x21a620dd2f6a53f3,
                ])),
                Felt::new(BigInteger256([
                    0xdbe578276a95cac4,
                    0xe0ccffa127596407,
                    0x86c38464985b489a,
                    0x18fc260f0792d182,
                ])),
                Felt::new(BigInteger256([
                    0xb8e299af8dbad1a8,
                    0xa59195c7932357e6,
                    0x1e1ebe28a42c88b8,
                    0x21a07b7b1a60b686,
                ])),
                Felt::new(BigInteger256([
                    0xcbcd50b55cdd338f,
                    0xc62c62c9ea160377,
                    0xf0e296aaee13fb6a,
                    0x37d8d2c4166f9561,
                ])),
                Felt::new(BigInteger256([
                    0x9e96e19caaf8a137,
                    0x67c3b762e0347eed,
                    0x06108bc3238eaf09,
                    0x28b3cfe5a56e181f,
                ])),
                Felt::new(BigInteger256([
                    0xa7cf5cdcf2beb289,
                    0xff774a47c48c9b1a,
                    0xfebeae124f32f8d5,
                    0x3f190ffb4af8d0fd,
                ])),
                Felt::new(BigInteger256([
                    0xda7fa6765aee758b,
                    0x89cd02e869719700,
                    0xeb6f329d81d0cbd4,
                    0x19a6ccf5381af479,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7b04332b9dfd02c2,
                    0x81cb39452555463d,
                    0xa5a99e012434f689,
                    0x2e63b785096edbec,
                ])),
                Felt::new(BigInteger256([
                    0xf17e5b4772aef12a,
                    0x90c1c1189268013d,
                    0xfa2fe360517e8de1,
                    0x1a4615e1d94ad601,
                ])),
                Felt::new(BigInteger256([
                    0x9785fc95cff63fe1,
                    0x64d62e2fc8ea7aac,
                    0x01537aa8db21aa0f,
                    0x0bc49eedc8007fb0,
                ])),
                Felt::new(BigInteger256([
                    0x25a592d8aad6a78e,
                    0x115ebefb1bf09fa9,
                    0x01320f4074ab9656,
                    0x356d0840b35873b8,
                ])),
                Felt::new(BigInteger256([
                    0x1b6259ee4841af34,
                    0x48e0ff2621036ee8,
                    0xa33daa105ae5f95f,
                    0x15adb1dd2c7e0ee2,
                ])),
                Felt::new(BigInteger256([
                    0x66bc835c425db812,
                    0xe842c39e4ba40bc7,
                    0x5327cd0584c90943,
                    0x3721b6f6bc51111f,
                ])),
                Felt::new(BigInteger256([
                    0xd4264cdd45226cf3,
                    0xb2f2dca7dcfd25f6,
                    0x72e31868307238f3,
                    0x1af9b7bb16b374d5,
                ])),
                Felt::new(BigInteger256([
                    0x17f2d8f9866fde69,
                    0x0bd73a2fdb83b12b,
                    0x0c95f54e5fc9f884,
                    0x0fe1f596d0360c73,
                ])),
                Felt::new(BigInteger256([
                    0xc135261ac42d6295,
                    0x0efa04efc94ddb29,
                    0xd3d7b86df0ef5f10,
                    0x320bc0b9bf6659a5,
                ])),
                Felt::new(BigInteger256([
                    0x9d6ad3b8ee5128a6,
                    0xe2b87724be373977,
                    0x0d4a57283a01e304,
                    0x19a3d69881f1e41c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa1e4324a0102e8f4,
                    0xfdf96f22bed0efec,
                    0x525303d0e2a78881,
                    0x1ab1a6a59c78232a,
                ])),
                Felt::new(BigInteger256([
                    0x5463d5f204e72377,
                    0x268cb94ef756f47f,
                    0x6fa703d74d92c0f8,
                    0x1c77737e832ca1ea,
                ])),
                Felt::new(BigInteger256([
                    0x1a58958e4af20855,
                    0x4f01ae1b92ed8b24,
                    0xbf6c43136e111090,
                    0x0ee123d303409f42,
                ])),
                Felt::new(BigInteger256([
                    0xa4e8271126dc3f07,
                    0x6e73d1ec7c283aef,
                    0x73d5cc3b716f4c66,
                    0x2d8231c251deffdc,
                ])),
                Felt::new(BigInteger256([
                    0xd32d415eddb3d7aa,
                    0x99027fe1725aaafc,
                    0x51feffe0cee92c88,
                    0x2b7ca709f5c2a508,
                ])),
                Felt::new(BigInteger256([
                    0x9da48463ae1246cb,
                    0xed1083c219829f15,
                    0x3fe6f613cae2ca24,
                    0x02033a744f36d747,
                ])),
                Felt::new(BigInteger256([
                    0xf5ec2fef717dcc05,
                    0x7ef8f8e45fce0e2c,
                    0x468f413e5a359694,
                    0x35491ea6977cd475,
                ])),
                Felt::new(BigInteger256([
                    0x7b66202054a2834e,
                    0x942eee300e24789c,
                    0x437fc4ccca2dbcf9,
                    0x2da880e682eb5764,
                ])),
                Felt::new(BigInteger256([
                    0xfe73ebe5b7bc3c3d,
                    0xc7d3de262479b969,
                    0x1c595b840b9db006,
                    0x133e203adab19d8d,
                ])),
                Felt::new(BigInteger256([
                    0x540b1da73c7d6b6f,
                    0x0ce7ba1abef8a825,
                    0x62c77f1449ef0c10,
                    0x03fac412a29f9307,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x597f69e707c2e483,
                0xbb37440063d1010d,
                0xea04eaca78a609e6,
                0x0849fbfff7e4804f,
            ]))],
            [Felt::new(BigInteger256([
                0xa68757198dce6cef,
                0x282f96e20a4ca4cb,
                0xe923d185d074cb3e,
                0x24452231e1e17dfa,
            ]))],
            [Felt::new(BigInteger256([
                0x114b408d1d58dc2b,
                0x57405dc6fc28eddd,
                0x17b357ba37cdf714,
                0x22e8234160bac5d1,
            ]))],
            [Felt::new(BigInteger256([
                0x814f15f37b3d5102,
                0x5ca501ed85084d53,
                0xa41ccb5788025712,
                0x1cb89f63c026ddf9,
            ]))],
            [Felt::new(BigInteger256([
                0xe6d0cd9fec228ef3,
                0x110f9f585c904fab,
                0xe93efe4af5dc8ff5,
                0x07527b5b9e741fab,
            ]))],
            [Felt::new(BigInteger256([
                0x1ea5230105f53222,
                0x9e5cdf76d8325bf0,
                0x9564052900597351,
                0x3a359335862945b8,
            ]))],
            [Felt::new(BigInteger256([
                0x128e8680781ddbb6,
                0x76b4229fd62fc342,
                0xd33503883358c713,
                0x2b66f2d5cc24a307,
            ]))],
            [Felt::new(BigInteger256([
                0x68d38904b13ba5d5,
                0x32e7b2dd90ca1c9b,
                0xe09c33e0e1f1869a,
                0x375e7389e63d90c7,
            ]))],
            [Felt::new(BigInteger256([
                0xfcef1b71501e2c62,
                0xc4f52df15ff5530d,
                0x10ba0bec76581cee,
                0x1331dcce589abe9b,
            ]))],
            [Felt::new(BigInteger256([
                0xbb4c6294f09a8f35,
                0xd56ce663933c2c0e,
                0xfaaee788240463ac,
                0x260c4eb881aebb0c,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 10));
        }
    }
}
