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
                0x298b27c186b34f0d,
                0x1c714553a6d7ecc4,
                0x57bea034d17715a5,
                0x696e559f184385eb,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x787a3b0eaeba0f02,
                    0x36b3ea8052648bd6,
                    0x67ed2e13fbff4c61,
                    0x34e65788d547b448,
                ])),
                Felt::new(BigInteger256([
                    0x97dd097cb22b0f6b,
                    0xe571fc2b233ef5f4,
                    0x83bdd56c7f9270bd,
                    0x5459f9f60a2ef707,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe165b1cf24a872d1,
                    0x8d55f1e5e081eef8,
                    0xaa305e8ae0c155d6,
                    0x4781b275746a3faf,
                ])),
                Felt::new(BigInteger256([
                    0x5f600bce85032daa,
                    0x220a0ea328999d3b,
                    0x83add7a9efb0d89b,
                    0x533afca723d10b23,
                ])),
                Felt::new(BigInteger256([
                    0x42be9accf6e26e66,
                    0xeb7a8da32828dbca,
                    0x0bcc62c571438e53,
                    0x522ab79cffd17c0f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x55dd76a162b3e96a,
                    0x63c5f2e88738412f,
                    0x7d300a28ecfc44b8,
                    0x3b478d62dd84ce71,
                ])),
                Felt::new(BigInteger256([
                    0x45792cba5002ee60,
                    0xa79e751473dcdc0f,
                    0x2b11a5cb57066602,
                    0x68f26b379ccd3bd6,
                ])),
                Felt::new(BigInteger256([
                    0xce24432f65522d1a,
                    0x17caa8a1014838e3,
                    0xc7e8c192c5f23b3f,
                    0x185da3501abcbe93,
                ])),
                Felt::new(BigInteger256([
                    0xdbb92263617af645,
                    0x26e697304bcfd110,
                    0xb522375212faf32b,
                    0x353de908e41068ff,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd0303def71f6436d,
                    0xbe6a973ae0dc45d9,
                    0x028ef695601a62fb,
                    0x66c678dc650d0785,
                ])),
                Felt::new(BigInteger256([
                    0x1139a4d899632cc7,
                    0xfc885bc321b51c83,
                    0x711ab57039800eea,
                    0x004d995dc3809c8f,
                ])),
                Felt::new(BigInteger256([
                    0xc23c35afe6dbddda,
                    0x50c5b7a0e873061b,
                    0x7064f1be22a10190,
                    0x501094e81bcdfae6,
                ])),
                Felt::new(BigInteger256([
                    0xd39cfa5883880126,
                    0x337b1228a8015d6f,
                    0xb8ec10470754e1d2,
                    0x1e345dfa12615059,
                ])),
                Felt::new(BigInteger256([
                    0x3692fef7550aee91,
                    0x5b0d59f6df5c569d,
                    0xef1f8fd12e605b67,
                    0x738ace72aae39c08,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5b858e7624ba319a,
                    0xd2a683e571910c7d,
                    0xe09609b925f45cf5,
                    0x38854d3bfc189dc5,
                ])),
                Felt::new(BigInteger256([
                    0x137a96b801ee06c8,
                    0x6fc9a2d0699f3ebb,
                    0x601dc4076fe6c15d,
                    0x6d97260f7b638c4c,
                ])),
                Felt::new(BigInteger256([
                    0xbf87622cc9b829a3,
                    0x3af6edd49122b833,
                    0x0d8ef885ebf2ddb6,
                    0x0c969f79369f6676,
                ])),
                Felt::new(BigInteger256([
                    0x6100deac79bcd357,
                    0xf8cbf10bb78a2123,
                    0x62da04e7719dc4cb,
                    0x559ccdd9e0223d8c,
                ])),
                Felt::new(BigInteger256([
                    0x408b4467b68b4cca,
                    0x656aa9f63d3d5f6b,
                    0xe23d54c0158577b4,
                    0x0831dffd2c9c66b4,
                ])),
                Felt::new(BigInteger256([
                    0xbc51dccbf8cc6bbd,
                    0x047cf3459b522398,
                    0x306982dd68f949ca,
                    0x56a35411d9c9717a,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xd9db286a5be7f696,
                0xeabd98d2261f9496,
                0xdf621fffab6fc090,
                0x0a85a261384cd18d,
            ]))],
            [Felt::new(BigInteger256([
                0x95a9fe435e6c8c48,
                0x40ade4158f10ab5b,
                0xd2b8802786602c55,
                0x09eaa1121fef8a05,
            ]))],
            [Felt::new(BigInteger256([
                0x5d8ab127a4e3ee00,
                0xfba8ecb65e7070be,
                0x140ef437cf79a382,
                0x1ba54562cf4447b1,
            ]))],
            [Felt::new(BigInteger256([
                0x16e51c240cc1a070,
                0xbecc1334de458373,
                0x917bda3acf2ef2a8,
                0x22714c8b0ed1ec6d,
            ]))],
            [Felt::new(BigInteger256([
                0x6d3db7ce0556a26b,
                0x87e105c3fce54c50,
                0xb0f847117500606c,
                0x5f332e0fbf0517a3,
            ]))],
            [Felt::new(BigInteger256([
                0x8afe92aefdb04874,
                0xc69a82435f3b15d1,
                0xae92b22074f2c08f,
                0x51159ff20251f968,
            ]))],
            [Felt::new(BigInteger256([
                0x50be81d033993683,
                0x22dd88e5980b8940,
                0xd79aa193dd127e00,
                0x672b72d360ad831e,
            ]))],
            [Felt::new(BigInteger256([
                0xd096e969228eebea,
                0x26022346f60db4a5,
                0x56236421fc5418cb,
                0x160281406d734cf9,
            ]))],
            [Felt::new(BigInteger256([
                0x1a2fa56f6cce6e3b,
                0x1d72f342ed33aecf,
                0xf8a12bd250383e54,
                0x2b4608f042894fac,
            ]))],
            [Felt::new(BigInteger256([
                0xe04ade350b972e87,
                0xab7226a080ebe881,
                0x12e7ea6efb0cb8ae,
                0x42efb9eb86613d63,
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
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xd9db286a5be7f696,
                0xeabd98d2261f9496,
                0xdf621fffab6fc090,
                0x0a85a261384cd18d,
            ]))],
            [Felt::new(BigInteger256([
                0x95a9fe435e6c8c48,
                0x40ade4158f10ab5b,
                0xd2b8802786602c55,
                0x09eaa1121fef8a05,
            ]))],
            [Felt::new(BigInteger256([
                0x5d8ab127a4e3ee00,
                0xfba8ecb65e7070be,
                0x140ef437cf79a382,
                0x1ba54562cf4447b1,
            ]))],
            [Felt::new(BigInteger256([
                0x16e51c240cc1a070,
                0xbecc1334de458373,
                0x917bda3acf2ef2a8,
                0x22714c8b0ed1ec6d,
            ]))],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 186];
            bytes[0..31].copy_from_slice(&to_bytes!(input[0]).unwrap()[0..31]);
            bytes[31..62].copy_from_slice(&to_bytes!(input[1]).unwrap()[0..31]);
            bytes[62..93].copy_from_slice(&to_bytes!(input[2]).unwrap()[0..31]);
            bytes[93..124].copy_from_slice(&to_bytes!(input[3]).unwrap()[0..31]);
            bytes[124..155].copy_from_slice(&to_bytes!(input[4]).unwrap()[0..31]);
            bytes[155..186].copy_from_slice(&to_bytes!(input[5]).unwrap()[0..31]);

            assert_eq!(expected, AnemoiHash::hash(&bytes).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
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
                    0x1851f75469f0db3e,
                    0x22e015267d584bba,
                    0x5f6e8a215bd01796,
                    0x57bc9cc7e15ab6fb,
                ])),
                Felt::new(BigInteger256([
                    0x173abade9dcc738d,
                    0x27038d54c5a3072f,
                    0x7e960de935dcb355,
                    0x52e0b3ff84bc9174,
                ])),
                Felt::new(BigInteger256([
                    0x377b80b2b1703e0c,
                    0x2b65c8a0e40c3883,
                    0x0d1741584ac074df,
                    0x4195f5c113420913,
                ])),
                Felt::new(BigInteger256([
                    0x22a7d057e455ac07,
                    0x2fd30031625556e8,
                    0xe44f951f508880d5,
                    0x56bc9596da7a57f2,
                ])),
                Felt::new(BigInteger256([
                    0xf37e16157885285c,
                    0x3c5e26557344702d,
                    0x68b52e5c10ddb079,
                    0x555b3bf269d36e96,
                ])),
                Felt::new(BigInteger256([
                    0xbd91f2cdbe2ca967,
                    0x0f2ce5d24a049c86,
                    0x27069c1ab3943363,
                    0x34026524f2fa3eda,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x96b0a8f869f4800a,
                    0xe3ce0c4f815a812e,
                    0xbe0e7f98a1d48679,
                    0x2485a551d10f5fdc,
                ])),
                Felt::new(BigInteger256([
                    0xc5ba44c06f65a201,
                    0x8a9146f6d5440518,
                    0x0604c37c2046afae,
                    0x322d88331557f3d0,
                ])),
                Felt::new(BigInteger256([
                    0xe095bf9bff8dec8c,
                    0x4cc12a3025599ba8,
                    0x2c3013328b61245b,
                    0x03b2a898a58f7780,
                ])),
                Felt::new(BigInteger256([
                    0x332f3718bc258fb5,
                    0xd76f1f953e6107cc,
                    0x524800661a7b5e16,
                    0x1c472c3f044a6dba,
                ])),
                Felt::new(BigInteger256([
                    0xc766094b973d49bb,
                    0xa41ca52e5c5d225e,
                    0xb7e2dcb038d2395f,
                    0x4be643d322945fb7,
                ])),
                Felt::new(BigInteger256([
                    0xbaea2413269ed55f,
                    0xde26f6c6ca99bc90,
                    0x5ec9da220e7000ef,
                    0x5afeb23914c7bed2,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5e2710c1e9b1e619,
                    0x959e14ac09a839bf,
                    0x706c83ca28830f65,
                    0x04c1d4992dab594f,
                ])),
                Felt::new(BigInteger256([
                    0x6f2da775311df1c1,
                    0xbaee50d01e7ee718,
                    0x8299e3215178832c,
                    0x55b17cba53572b96,
                ])),
                Felt::new(BigInteger256([
                    0x54f41f403b0a5a29,
                    0x14c787e74412bbc8,
                    0xa391fb5392a727e6,
                    0x6087d48a7325f0ab,
                ])),
                Felt::new(BigInteger256([
                    0x249967c75a91d3dd,
                    0xd72d9c38f639f1bc,
                    0xc6fbd2ff174d4329,
                    0x2bc8490e4b8f566b,
                ])),
                Felt::new(BigInteger256([
                    0x096b3b05fe846368,
                    0xb8448001b84a8382,
                    0xc83ad0a92ca99aad,
                    0x3a091022d123540f,
                ])),
                Felt::new(BigInteger256([
                    0xf3213aeebf5a7cf6,
                    0xb1d882f900caff35,
                    0x9fe3ba9eb3fd55b6,
                    0x36b0fd753797f982,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf7a9e6e058c10d7a,
                    0x9887b45a1b461c1e,
                    0x05e81b796b6bfb3f,
                    0x0da752786177985b,
                ])),
                Felt::new(BigInteger256([
                    0x5aed43385975c177,
                    0xe57a7f51c8076eb9,
                    0x9cab54dc77cf42f0,
                    0x58443f212d462124,
                ])),
                Felt::new(BigInteger256([
                    0xcf83352858cab848,
                    0x6f0d73fcc57cea19,
                    0x811f5dd7cf319e4d,
                    0x6624d09e6d677ab0,
                ])),
                Felt::new(BigInteger256([
                    0x7ad25d77edbaedb8,
                    0x1cd07cee71b620d5,
                    0xf65a9e9fa24eec30,
                    0x452d9f0372b5a45a,
                ])),
                Felt::new(BigInteger256([
                    0x01c6f4413c2ca57b,
                    0x3615c2263e6f5cfa,
                    0xcc774fcf64873a12,
                    0x0e415d17b53e7c0d,
                ])),
                Felt::new(BigInteger256([
                    0xd42c16ff947854a7,
                    0x28ba65557d1d39f7,
                    0x1fa288c52148b7d9,
                    0x0eba74aeaa2c7ab6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xbb97cc4ffed63d17,
                    0x24aae22e8a0dd6ab,
                    0x747f5db2008648fa,
                    0x1ae3513447027fcc,
                ])),
                Felt::new(BigInteger256([
                    0x529a61a78abdf3da,
                    0xdfd84f5aa179fdaa,
                    0xc323efab4c342ee9,
                    0x130bfeeb76021602,
                ])),
                Felt::new(BigInteger256([
                    0x8a813ac4b83555cc,
                    0x98d20b4c9108af60,
                    0xf3a63befe7625106,
                    0x38e5581ecd128abb,
                ])),
                Felt::new(BigInteger256([
                    0x6a11b2c702247974,
                    0x3f634de219d8a921,
                    0xa9c9539aa2133e1f,
                    0x57dbfc9d3911065d,
                ])),
                Felt::new(BigInteger256([
                    0xff29139bdad436a9,
                    0x0111f9a00b85b49d,
                    0x2c468722ed7aaf09,
                    0x2dd247dfddad19aa,
                ])),
                Felt::new(BigInteger256([
                    0xfc46d84238426749,
                    0x970efeaafa96247d,
                    0xbad0badd02e760fd,
                    0x3add3f9459f1eb16,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf4780a3f440eb0ee,
                    0x34cc0054b98f9210,
                    0xaa68693a2f98e667,
                    0x515e8287036a6555,
                ])),
                Felt::new(BigInteger256([
                    0x3b1205df857c4b2c,
                    0x63d7751c0c413b1a,
                    0x7eae4dad98988e88,
                    0x1a695fa036b3b34e,
                ])),
                Felt::new(BigInteger256([
                    0xbaee16a4b9e10e2c,
                    0xfccdcdb3a8e17760,
                    0x2be4c908536d7fab,
                    0x5fda5d23bd5e5d87,
                ])),
                Felt::new(BigInteger256([
                    0x450f4540b21d598d,
                    0x798daae252092e16,
                    0x54b8bb95536d6e6d,
                    0x3bb8aa72b27bb83e,
                ])),
                Felt::new(BigInteger256([
                    0x9fe53af1d1e90eb2,
                    0xd9b6ae0f9e53cfbc,
                    0x8b8f38c6e675c489,
                    0x5f614048e2c034c8,
                ])),
                Felt::new(BigInteger256([
                    0xdf7c794a811f1145,
                    0x0dfbb6340ebe96fb,
                    0x72900d70e8d76dff,
                    0x3de8ca557c46dee7,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x8614ec4a53fd295d,
                    0xba39bcf85db8bd3a,
                    0x2674dad11dde33cb,
                    0x309f855d5725df4f,
                ])),
                Felt::new(BigInteger256([
                    0x869c67e3a635a529,
                    0x0fc496bea613a24e,
                    0x17caea9677030b96,
                    0x3f87fed80395e123,
                ])),
                Felt::new(BigInteger256([
                    0x63c0fe1a786425df,
                    0x959591100607ffce,
                    0x9b7918675e6fd44b,
                    0x35af4479c228b2a0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xea437d0627819b07,
                    0x35a4a0f534d4e7ec,
                    0x7205a73f339e21f1,
                    0x43e546c8c5b33d0c,
                ])),
                Felt::new(BigInteger256([
                    0xf1ea366f147e6b0f,
                    0x5d7c3ebff7c084e4,
                    0x4abaa40d0f207f4b,
                    0x56796ebda3f0a840,
                ])),
                Felt::new(BigInteger256([
                    0x7b920fe7c65c308f,
                    0x280874c7424d1559,
                    0xa6d56844129283f6,
                    0x1caf4cfafcc241ca,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9fdb9eaac8c9da71,
                    0x7aaeb333d7085562,
                    0x1c709f49ee2294ec,
                    0x2884a588a6ff51d3,
                ])),
                Felt::new(BigInteger256([
                    0x9dd288152f9fa7a2,
                    0xa567030869ecce37,
                    0xcecb5c978b4e86f9,
                    0x019e073d7c0ddc22,
                ])),
                Felt::new(BigInteger256([
                    0x04bc053a5a6148c8,
                    0xed0e01472e1cc839,
                    0xe846bdc5abc75dd1,
                    0x423db3769f38067a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x286bbb3446706023,
                    0x84dcca7370bafed2,
                    0x7843f376baaed9dc,
                    0x44c4c26432c6173e,
                ])),
                Felt::new(BigInteger256([
                    0xd1f67bdbaa17a781,
                    0x76e60faefe2546a6,
                    0x32ac4929b4825ef5,
                    0x61b10b4f3ac8ed8e,
                ])),
                Felt::new(BigInteger256([
                    0xe9fa714857ab32be,
                    0xa2e5bb2058756ae3,
                    0x4c9f4cec7b956dae,
                    0x3100b2ad3df69374,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5e8806a4de95264b,
                    0x4f572c6f2db82277,
                    0x96a0638d2a60555b,
                    0x6e938ee24b7ee751,
                ])),
                Felt::new(BigInteger256([
                    0x3d7ce67729ebc292,
                    0xae018c90cd6e19ce,
                    0x8d374b849a087cc5,
                    0x259108ed618779d5,
                ])),
                Felt::new(BigInteger256([
                    0x333744b9e3a85f16,
                    0x7c7f1fe85d396461,
                    0x7f4aca0ff16b3cc9,
                    0x4d543eb3d0be9ab7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8e982dec1c39bd4f,
                    0x5b4e840d8a0a22fa,
                    0x4b1ca100886fbf15,
                    0x2bd77965c55f68b7,
                ])),
                Felt::new(BigInteger256([
                    0x94165746350ce84f,
                    0x41f383f405bd7ce6,
                    0x0174dbf6db07877a,
                    0x6b599b8ee4d7fbd5,
                ])),
                Felt::new(BigInteger256([
                    0x556323ecb5b31452,
                    0xf67699f4f2b5cc6a,
                    0xe59b97a57adc320a,
                    0x6cd2eb17563c4ef5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd347f881aa68f434,
                    0x33ac0db6b703f83a,
                    0xe116aff40a2f93a8,
                    0x6b1ebe2f7cc259b0,
                ])),
                Felt::new(BigInteger256([
                    0x7a11adb64496a2d7,
                    0xd450b1c2927291c4,
                    0xd0215e818dd3562b,
                    0x4640fb8c4f326911,
                ])),
                Felt::new(BigInteger256([
                    0x583bb76762d7b5b7,
                    0x63dbc1d279dd35db,
                    0xb41fb016fc9e0b9d,
                    0x334682b296b54537,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0424c8d96f46b299,
                    0x5627059af9f90d42,
                    0x338f8e4ad4a11c06,
                    0x26f26a727ee3dd33,
                ])),
                Felt::new(BigInteger256([
                    0x68403e57eb671648,
                    0xb87eca98d3659080,
                    0xe1d974465241f7c4,
                    0x56081076a7f3917b,
                ])),
                Felt::new(BigInteger256([
                    0xa764efb372f6e656,
                    0x9a3d75585d1ea6fa,
                    0x3ec6af7adea47996,
                    0x5f1bc80b1bf82016,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa3528d8af68a0393,
                    0xece80c6962a12f5b,
                    0xa41f7e48538de22e,
                    0x21ba9049bea7fb94,
                ])),
                Felt::new(BigInteger256([
                    0xba463e8e9f6d6105,
                    0x9508f8a4c786cc9d,
                    0x3ede9d288415453d,
                    0x56193658d2d4616b,
                ])),
                Felt::new(BigInteger256([
                    0x48b7d069b5de7921,
                    0xd1cc3168c4c345a3,
                    0x47695f3116bcb2d5,
                    0x649eb7d902d7a0ab,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x51dbe3481d6495ca,
                    0x995d60db4cdf717e,
                    0x25b1411c06c783d9,
                    0x19701075f44369ac,
                ])),
                Felt::new(BigInteger256([
                    0x85bc597820a232b9,
                    0xe0d345c2ebe0e1f9,
                    0x535789167a918883,
                    0x1c11569ee7b8a6de,
                ])),
                Felt::new(BigInteger256([
                    0x267b278762fdf843,
                    0x38101a01f903cff8,
                    0xc4f1a5fcb69e4335,
                    0x24b3263b75fad2a8,
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
                    0xf75fbf4bb913647a,
                    0x6b587f50ffdedcb2,
                    0xfaf44aa9c34096c3,
                    0x01dc87eec3521f9e,
                ])),
                Felt::new(BigInteger256([
                    0x5c7b8feaff3c0ce4,
                    0x7e314a8c73d90fc4,
                    0xe01266b3160831b8,
                    0x19fe72835f4cbdf2,
                ])),
                Felt::new(BigInteger256([
                    0xe00b4ee4ed4cf09f,
                    0x39d0af0191760f24,
                    0x0bcea8514223e8fc,
                    0x5aa71f5abaf7297a,
                ])),
                Felt::new(BigInteger256([
                    0x8b4ecc7b56e9170b,
                    0xe3cad55f0e5b16ff,
                    0xed943afa8e19997a,
                    0x03aa6cee09fda505,
                ])),
                Felt::new(BigInteger256([
                    0xecaad23ccea0f774,
                    0xd63f25b5568010a7,
                    0x4b05d24bed69a167,
                    0x032ce8e3ef6fe8b2,
                ])),
                Felt::new(BigInteger256([
                    0x19b617ddb6ab0fcc,
                    0x9ef0cb6bd7d29f5e,
                    0xe28f58bd0c6896b9,
                    0x6cf7dd9d221e86bf,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf48c05a21b92545d,
                    0xe84ecd452b4f7930,
                    0x23d996db86ab60c8,
                    0x0cd04d073c4255c8,
                ])),
                Felt::new(BigInteger256([
                    0x81de2d1ad9fe9451,
                    0x7bc8238cba63169b,
                    0x5a3eab3c30cceac5,
                    0x0ff957e46123be9f,
                ])),
                Felt::new(BigInteger256([
                    0x0e4c216f80448e82,
                    0x2fc23a1daad9f983,
                    0x7470206f5a921323,
                    0x66ff83049bfb8bd5,
                ])),
                Felt::new(BigInteger256([
                    0x8dca1d10556dd69b,
                    0x830f5034b160b0e4,
                    0x0741612ab1efa500,
                    0x5a5c8d6b4b8e59e9,
                ])),
                Felt::new(BigInteger256([
                    0xc25e2b068712268e,
                    0xf960102f6f64072b,
                    0x8c9c5585300df1f9,
                    0x4a24f4561952817f,
                ])),
                Felt::new(BigInteger256([
                    0x7b0aed3f959d93b7,
                    0x7e76d7e5d03f7fe6,
                    0xee339f91f1acb3bf,
                    0x25e287def3f0ab68,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x33fc479f775b9007,
                    0xbad2281eb41a54b0,
                    0xd41216b53c638a9b,
                    0x63436406e33b6db4,
                ])),
                Felt::new(BigInteger256([
                    0xdb0b92a77d754124,
                    0x8273feaa7443eee6,
                    0x759051d4565cd74b,
                    0x4b6c97b2c2bbf125,
                ])),
                Felt::new(BigInteger256([
                    0xa022688a9f1e80ce,
                    0xdb5b0a10965d7592,
                    0xe19fe38fb9a7619d,
                    0x12139b25046bf072,
                ])),
                Felt::new(BigInteger256([
                    0x7033b0760f87fe40,
                    0x08dde8ca383d1095,
                    0x4600c3763df5e6b0,
                    0x6c2533e8be1278ee,
                ])),
                Felt::new(BigInteger256([
                    0xe19cecb37269711a,
                    0x59fb5dfa64f0c99f,
                    0x7fbbc1b1df33ec3a,
                    0x26ecbeee902fdba2,
                ])),
                Felt::new(BigInteger256([
                    0x5ac71e69e5900b40,
                    0xbd7b16587a39f9e1,
                    0x5442b601242164d4,
                    0x6848c0f59346c866,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0270615989440c28,
                    0x89c7d5334a26e7af,
                    0x3084373260e4e572,
                    0x72d2124b4d2413e1,
                ])),
                Felt::new(BigInteger256([
                    0xbb74c06fc295d81e,
                    0x2610cb4305a5458d,
                    0xc86b0e30185aa0ce,
                    0x542edacc97a5ee31,
                ])),
                Felt::new(BigInteger256([
                    0xa2fc0d8a98e20657,
                    0x6241aea0cc01c36a,
                    0x0bea603567c0f6fc,
                    0x678e0b49eb290a04,
                ])),
                Felt::new(BigInteger256([
                    0x8e2d64726421a836,
                    0xab2664255a1c381d,
                    0x12dec711aa9c0ec6,
                    0x056ba067fcc61e7a,
                ])),
                Felt::new(BigInteger256([
                    0xeef517bb83d1a9a8,
                    0x5f1d01f3debe92af,
                    0x8aebc89b1173184d,
                    0x72d92171428343a6,
                ])),
                Felt::new(BigInteger256([
                    0x1d12ba18f0b4dec2,
                    0xa7c603f06f744664,
                    0x6ad92775d090c584,
                    0x444e4d6d3a0ab67b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x4ef65484295d588c,
                    0x81babd71320ae92d,
                    0x43b0ce617a118401,
                    0x6ff11e1165bddbda,
                ])),
                Felt::new(BigInteger256([
                    0x600cebd425e8b983,
                    0x05840259b5d6fad2,
                    0x86ef402681ab31a6,
                    0x4e7766027bbe7752,
                ])),
                Felt::new(BigInteger256([
                    0xd47cae7b1bbc9e44,
                    0xbea25b67e362964b,
                    0x76b12addc297e596,
                    0x0ee84ae5d3464a01,
                ])),
                Felt::new(BigInteger256([
                    0x6c9447c5ef19ffa7,
                    0xbafd4be0fc30609d,
                    0xac9fad6f72590318,
                    0x26a8281add7266fd,
                ])),
                Felt::new(BigInteger256([
                    0x2c1c61d8e4c50f8f,
                    0x13f17b2631f2ae1c,
                    0x844fe9b64b00816d,
                    0x3d91ee97c7a39e4a,
                ])),
                Felt::new(BigInteger256([
                    0xf635e20cfc5a6d93,
                    0xcbeb25e0362c4e98,
                    0x63731664909727d9,
                    0x52fb2c7a32cb67a4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x79e07c2e1001a30e,
                    0x60319387b84b525b,
                    0x32642902dbf407ae,
                    0x5d5d7cbe95d24e8f,
                ])),
                Felt::new(BigInteger256([
                    0xd385fea20dbc0ef1,
                    0xe79c4a224d268113,
                    0xccd63fbd7268b741,
                    0x1b1c14048329925f,
                ])),
                Felt::new(BigInteger256([
                    0xf4ad7c834f9b3e62,
                    0x21318b49c4aed857,
                    0x30ca4ff4b408106c,
                    0x428b66b1b9ebb462,
                ])),
                Felt::new(BigInteger256([
                    0x2e82e0c4f91c210c,
                    0x8bd1821e3b1a311e,
                    0x8b9f99c6df74e409,
                    0x710ed644e5ed06fe,
                ])),
                Felt::new(BigInteger256([
                    0x7ad6e0dea49f5fc9,
                    0xd436530c1e3a9007,
                    0xd8357e95356265df,
                    0x2a5acbdfcf328a19,
                ])),
                Felt::new(BigInteger256([
                    0xa2f7943c651d3c56,
                    0x95e5039723a15c46,
                    0x1775811cae1de7e9,
                    0x7108048c56d8ffa4,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x707252497296f464,
                0x0bd640c409d60358,
                0xa67f05c6e9af3ba8,
                0x31e9215bf346f5ca,
            ]))],
            [Felt::new(BigInteger256([
                0x57bfc35e025c36a4,
                0x676bb0796ee4262c,
                0x305bdb884baf4d2d,
                0x43205b2e3cc8a9cf,
            ]))],
            [Felt::new(BigInteger256([
                0x426a2bfa52cacadb,
                0x0d23b7836f11ebd3,
                0xd382b9a7253879b8,
                0x6c60603cc2453470,
            ]))],
            [Felt::new(BigInteger256([
                0xe45ca85948333a61,
                0x4aeaf13fc757545d,
                0xc455b184e124ce7b,
                0x6388d90d81e81af8,
            ]))],
            [Felt::new(BigInteger256([
                0xd6e5a65cbea1fea4,
                0x33d4f8622670ae41,
                0x2dc339677cfefcfd,
                0x717ce11b3fa9fad6,
            ]))],
            [Felt::new(BigInteger256([
                0x0bfd9c79c1262c71,
                0x0374923c681bebca,
                0xeb4c9f9a6188e8d9,
                0x59246909ecd7c0cc,
            ]))],
            [Felt::new(BigInteger256([
                0xa1ce89369f2f3b02,
                0x3355249a297b2e4e,
                0x864929384d245ac1,
                0x0ff6bb489bf05461,
            ]))],
            [Felt::new(BigInteger256([
                0xac35197627cd9a97,
                0x23695d4aaed3bf1b,
                0xc83b45a663c332ea,
                0x593289b2522f897c,
            ]))],
            [Felt::new(BigInteger256([
                0x3fd9f3e4e7eb169b,
                0xdebd2d9130670c2f,
                0xba41ed8cf686c306,
                0x0ddc09ad62e918cc,
            ]))],
            [Felt::new(BigInteger256([
                0x4dcde418feee9172,
                0xcdd0d978595ca200,
                0xa371ce8de090bf55,
                0x28f8e85f5527bce8,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 6));
        }
    }
}
