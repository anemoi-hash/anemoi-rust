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
        // We can output as few as 2 elements while
        // maintaining the targeted security level.
        assert!(k <= NUM_COLUMNS);

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
            vec![Felt::zero(); 8],
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
                0x57c860cb42bb6262,
                0x9c4f8283eec66fd9,
                0x543a2948740e8669,
                0x219fe9cb6b382ca4,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x632764ceff310622,
                    0x62a07fc47ee84872,
                    0x78584478b611e990,
                    0x3c03cd0c58c7d6c4,
                ])),
                Felt::new(BigInteger256([
                    0x6463f13e31f2cdd8,
                    0x9358a53270520d09,
                    0xa1012a6719dbeaec,
                    0x14348dc26714ef59,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf9cac876c426295c,
                    0xfad64dc63ed1b4f0,
                    0xb7a0db20562cd381,
                    0x3fabeb96a7dba291,
                ])),
                Felt::new(BigInteger256([
                    0x4fa78581071c8a9a,
                    0x459c20e5d855c44b,
                    0x10cff468ddb43313,
                    0x30b105ff8dafd943,
                ])),
                Felt::new(BigInteger256([
                    0x1abab1ab03fd3457,
                    0x33fab5f88b791218,
                    0x8f64a5dfe5e3c9cb,
                    0x2acc374b5070338e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xca79d32909a23201,
                    0x7b012774a4c9d45e,
                    0x09fba49004fe6f50,
                    0x236fd5fad2e5ff3d,
                ])),
                Felt::new(BigInteger256([
                    0xc41c0d43553d4232,
                    0xc66873fbcc5a76e7,
                    0x2e6a9429757a8e1b,
                    0x09982c4a10cd052b,
                ])),
                Felt::new(BigInteger256([
                    0x2312d6c6eb95a45c,
                    0xdac149a6a89430a5,
                    0x3db8313199b2e7bd,
                    0x0b80e331415a55c9,
                ])),
                Felt::new(BigInteger256([
                    0xec2093a86d85c054,
                    0x4a09861531dbeba3,
                    0x631f082419e5c44b,
                    0x2fb49babbfc8ad1e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa6fc1a4ff9797cbd,
                    0x786a5206ee0aed51,
                    0xdc04bb18008416ee,
                    0x330cc113cad349d5,
                ])),
                Felt::new(BigInteger256([
                    0xb9d53185e1cd5c6d,
                    0x8ce9df605a04f1e9,
                    0x772ce35d7598b880,
                    0x356283b296c9d83b,
                ])),
                Felt::new(BigInteger256([
                    0xe344a3982b22ce72,
                    0x760120740cbe6a8d,
                    0x1db72913a8dac57d,
                    0x3ae787981fb8ac13,
                ])),
                Felt::new(BigInteger256([
                    0x3af43d5a61c6b79f,
                    0x5d84fa7d1c36330a,
                    0x622531b5b63bac14,
                    0x1656206428bac34a,
                ])),
                Felt::new(BigInteger256([
                    0xb8c32afbaf1a638e,
                    0x73d0817f5a03e9dd,
                    0x219f3d33ee2d30c1,
                    0x1c1de7acc368cce0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3999a7011a81a1dc,
                    0x99f8180173fd958f,
                    0x416d1c2f155501f4,
                    0x30ec6f8cb7182a87,
                ])),
                Felt::new(BigInteger256([
                    0x66e6cc52ed79fcf1,
                    0xd0eb1774f0449185,
                    0x158a2f29a8abbd00,
                    0x035a4fda5a7e18b3,
                ])),
                Felt::new(BigInteger256([
                    0x0a5c180a8f4e8451,
                    0x397976c442dc820c,
                    0x4f57407631cce9fc,
                    0x11eb08e3c7162865,
                ])),
                Felt::new(BigInteger256([
                    0x17eb669bb9744ce2,
                    0x1e07e7eaa39ddf2d,
                    0x257360a6ddab20a7,
                    0x3d554afa21d4763e,
                ])),
                Felt::new(BigInteger256([
                    0x15689c5c3cbe6b11,
                    0x53986e24b35c22eb,
                    0xaace86614376f0a6,
                    0x3657d2578728c5c9,
                ])),
                Felt::new(BigInteger256([
                    0x471788425dcc8da9,
                    0x5a50398b4e091c5f,
                    0x712022304fb42c82,
                    0x3f58f99efb4a2649,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xcc224680a5951c81,
                    0x18816b088f77ae6d,
                    0x0bb7896612ea7c3d,
                    0x21d93fe2a2934cdf,
                ])),
                Felt::new(BigInteger256([
                    0xa3a317482aa9f2b3,
                    0xe39fa625b70e4ede,
                    0x82d46eede7354667,
                    0x2d55f026ef4b39ce,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9f8491ab0dddbef7,
                    0xd7df2dded25c3971,
                    0x924ca8f9dcf7dc40,
                    0x23ef2ff2e6077c41,
                ])),
                Felt::new(BigInteger256([
                    0xd7830d35acfe1006,
                    0xc82ef5de82d7f43f,
                    0x0552df09f4b76c4c,
                    0x3c0a31b702678844,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xcfbaac09c1ce05c7,
                    0x1a6beada0db8203a,
                    0x066e742ec0a52db2,
                    0x28320f36e162e5aa,
                ])),
                Felt::new(BigInteger256([
                    0x66f7824afcbcc3dd,
                    0x3cf53f8abdad4654,
                    0x5e708eea40225763,
                    0x147bcdb2c782e5ab,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9a0eda59f950b948,
                    0x6641f9a17ab17216,
                    0x039166b17f27c6e4,
                    0x1f3c6c3b999a264e,
                ])),
                Felt::new(BigInteger256([
                    0xfbff92c47d44d9ad,
                    0x59fa72e6292ced14,
                    0x78dff531701f6781,
                    0x0701791b04aa4165,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x36e0dea986836d89,
                    0xc2323e32c8ca015a,
                    0x80d4c2a14d2bc09c,
                    0x2d0cdb41c5cb0e1a,
                ])),
                Felt::new(BigInteger256([
                    0x7059bf38b9ac3147,
                    0x0d5a94bafd247d99,
                    0x236b9225046c6c80,
                    0x22f3b2b7e448c814,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbb98f33cd1aeca90,
                    0x9b5c3bc37c3aaa95,
                    0x3a3374d132716197,
                    0x33e8d5380c95d2ca,
                ])),
                Felt::new(BigInteger256([
                    0xf44e7bb11f993c8e,
                    0xa7b1bfac92afaf85,
                    0x0a0f2c377b1ba82c,
                    0x2945ac07a42e446b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7e43ef1d188514ff,
                    0xb07cc8941a4b5255,
                    0xb0ea9cf0ab4471a6,
                    0x38d565609ef5b69d,
                ])),
                Felt::new(BigInteger256([
                    0x871ed4b102685ef6,
                    0x9d15f3916fd0c6a7,
                    0xea4cbd8b6dad0373,
                    0x3cfc55319dead7b4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9969df9d1bea9030,
                    0xb300148cb1fc147b,
                    0x261164a1e3bf9c31,
                    0x29aef7a762c38090,
                ])),
                Felt::new(BigInteger256([
                    0xb06490c68d16154d,
                    0xde185c0a799c60db,
                    0x87da8a2e70df8617,
                    0x09f7630a5743506b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd5cf62298bc7a279,
                    0xc78e938f19087dbd,
                    0xc341d2331eb6edcd,
                    0x1c53c6c9b028a521,
                ])),
                Felt::new(BigInteger256([
                    0x4c60ef393cc802a1,
                    0xdf7d70296f29ecd1,
                    0x4b59cfc100ff134f,
                    0x258041eee73a1707,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x00373e07851759ea,
                    0x9de1a237c4d988b0,
                    0x092ea5bce1734161,
                    0x0b5d7f4e91d72f73,
                ])),
                Felt::new(BigInteger256([
                    0x3cd2834c1f221b7b,
                    0x087a5c0e978deb05,
                    0x74dc72cfc5b22ecb,
                    0x090dade3ccb792aa,
                ])),
            ],
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
            vec![
                Felt::new(BigInteger256([
                    0xdb49fb1506c34a0a,
                    0x170e44bea64688df,
                    0x48169d4415d9ee8d,
                    0x1a18d76154f26aa1,
                ])),
                Felt::new(BigInteger256([
                    0x4a9174f8a8ae26ce,
                    0xa28dd026bff6ce16,
                    0x6f05b7796d3b85fe,
                    0x2a1f12e34078ebc5,
                ])),
                Felt::new(BigInteger256([
                    0x378a22a772fa5071,
                    0x70b278eb5ada6324,
                    0x700a37cb5a7624cc,
                    0x3169e50b15b437b3,
                ])),
                Felt::new(BigInteger256([
                    0xaa5d51f50cd2c1c6,
                    0x2cfc7b9eba27dad4,
                    0x4d485b4303b2276a,
                    0x38989066f71b24bf,
                ])),
                Felt::new(BigInteger256([
                    0xd03cf18254fbdeb4,
                    0x78cc2561e12d2975,
                    0xc740a2c9bd10f22c,
                    0x02eac16bc97dad9c,
                ])),
                Felt::new(BigInteger256([
                    0x9144079d76f46cb3,
                    0x0c9861b432d08563,
                    0x427dc30b53ad6a4d,
                    0x157f0370fcc83491,
                ])),
                Felt::new(BigInteger256([
                    0xa2bc1842d1e46e8c,
                    0x77df218c0bfeaf5f,
                    0xd01415841a89c40b,
                    0x3ad404612d72d781,
                ])),
                Felt::new(BigInteger256([
                    0x00ffd6b318b0078f,
                    0x3d67ae8bc1909145,
                    0xbd64f5291f1d97b1,
                    0x03bb5c23e749b79b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7a2d7df620955f50,
                    0x31a5e8bddc16e359,
                    0x0166b1fa33943558,
                    0x35093bee3db63792,
                ])),
                Felt::new(BigInteger256([
                    0xa1c16917a2531dcd,
                    0xc31206408058802f,
                    0xda8a963128e5ec47,
                    0x10e10e4c78da5c99,
                ])),
                Felt::new(BigInteger256([
                    0x0a19f62de48ed97b,
                    0xcac28c3f72d3c045,
                    0x478434b3ab6241f7,
                    0x24b228fc380229ec,
                ])),
                Felt::new(BigInteger256([
                    0x8fe0204c40c39f4b,
                    0x179931bd0d398502,
                    0xbc6928335c371677,
                    0x20a65fd359e7fc62,
                ])),
                Felt::new(BigInteger256([
                    0xda4da982035a8611,
                    0x0a271f6e2fb533a6,
                    0x2cb0f17216711482,
                    0x01cf3c3cf139c4b2,
                ])),
                Felt::new(BigInteger256([
                    0xe80bdfa21f0d64c8,
                    0x7cd65fa45e95370a,
                    0xcd9a619a11ae5ae9,
                    0x3aba4264ece9a40a,
                ])),
                Felt::new(BigInteger256([
                    0x315fd0707038f434,
                    0x57441c95c99359e6,
                    0x1ea3d9c451c1662f,
                    0x01a50d6172d8120f,
                ])),
                Felt::new(BigInteger256([
                    0x7b7ab0b08d5d4c0a,
                    0x47bd2cdee2c36c03,
                    0x28709a6792c8981c,
                    0x3f3e2fc671ad368f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x25a081e67a14fc1f,
                    0x443dfedb70883f6f,
                    0xb18c25f49853dc56,
                    0x1e5ec47179fed78d,
                ])),
                Felt::new(BigInteger256([
                    0x10726c23659d4368,
                    0xf7a328e48de6e0b9,
                    0x43cb056ae899a378,
                    0x3a669c9584e322bc,
                ])),
                Felt::new(BigInteger256([
                    0xaed7c15880211c79,
                    0x56c59bfab117f5a3,
                    0x79fa259c1d765c5b,
                    0x1224685c731c341b,
                ])),
                Felt::new(BigInteger256([
                    0x3e275b5f46042419,
                    0xec5d8cf91a1251cc,
                    0x41a232f8d6265693,
                    0x272a2889e38f8bf6,
                ])),
                Felt::new(BigInteger256([
                    0xbbeec42cb9830b4c,
                    0xe52fe5bc9d4dcfb2,
                    0x1e58714b3008c709,
                    0x0e71f891e12f3a6d,
                ])),
                Felt::new(BigInteger256([
                    0x1dbfff03581b2e57,
                    0x274f2da1a7661f94,
                    0xcecb0104352a70da,
                    0x0c00c3e330873301,
                ])),
                Felt::new(BigInteger256([
                    0xf83be453d0a8fb8d,
                    0xdb6fbd843a2e8316,
                    0x895aeb87f8e5962b,
                    0x216467b9d3eec05c,
                ])),
                Felt::new(BigInteger256([
                    0x01f1d7b9d8caf305,
                    0x44f041cde36b2047,
                    0x5a74f1ed07673fa8,
                    0x38cc53b3ea26f519,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x1334521db19e38e0,
                    0x7082785cc119e7e4,
                    0xfa84e3399ed12358,
                    0x333ab29b26fab90f,
                ])),
                Felt::new(BigInteger256([
                    0x2b59ae47847a388e,
                    0x36772127a578bda8,
                    0x347edc42cc6d2496,
                    0x268af766e16ad861,
                ])),
                Felt::new(BigInteger256([
                    0x369ea792d6fb67e2,
                    0x621efd23d629c21c,
                    0x2ca08c92865326cb,
                    0x1e25e1874bb8ec5b,
                ])),
                Felt::new(BigInteger256([
                    0x1a75eb497ef43eba,
                    0x05c1489abcd1688e,
                    0x871e35b54f56a250,
                    0x212a6b9d75117867,
                ])),
                Felt::new(BigInteger256([
                    0xf57f383aa49801b5,
                    0xdb8266072ae710a7,
                    0xef84fc2622fa3f89,
                    0x27570cdbc1e20255,
                ])),
                Felt::new(BigInteger256([
                    0xe9e2306b259e97ad,
                    0x854986ef349dc685,
                    0x7f187e2b2dd3ed3d,
                    0x369a6703fec8286e,
                ])),
                Felt::new(BigInteger256([
                    0xb8a38c62f1f7c2ce,
                    0x380073cc5e72aec4,
                    0x1ac074495c8cb7b2,
                    0x1fbe8dd0317c51cd,
                ])),
                Felt::new(BigInteger256([
                    0x62f51c7c705fdffe,
                    0x9ebd1b7b1e4d387e,
                    0x4872870f897cec7f,
                    0x1fab5526fd0f9948,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xbbb6e889ef00a602,
                    0x4744781f70b893cf,
                    0x4a664090305bbc48,
                    0x1a6d464e5c3fc00c,
                ])),
                Felt::new(BigInteger256([
                    0xb2dd940ae8b006f4,
                    0x6813469c96cda1a8,
                    0x1e2077101ccf5760,
                    0x20a61cd5bbf1daa6,
                ])),
                Felt::new(BigInteger256([
                    0xd31c938e503e0abe,
                    0x1c4153e7004d53fc,
                    0xa34d2990f1905ba3,
                    0x11b81748816130ea,
                ])),
                Felt::new(BigInteger256([
                    0x12b5e3527c37f99d,
                    0x4eb1993421f1f8dc,
                    0x19dece89d2d15996,
                    0x0646f9db0c1f25d3,
                ])),
                Felt::new(BigInteger256([
                    0x24ae21ee95d484f8,
                    0xaa96c04e89ca7222,
                    0x17a60a604684cda4,
                    0x11e597fbe9b10675,
                ])),
                Felt::new(BigInteger256([
                    0x24cf0fca4ca36647,
                    0x89b54cebf7bb9356,
                    0x9f13890957cd7a9a,
                    0x37780be38f8d3151,
                ])),
                Felt::new(BigInteger256([
                    0x26483f32c6f256ba,
                    0xa354fb3b60d5bff4,
                    0x92277bd99bb416c2,
                    0x0e676431bac93fe1,
                ])),
                Felt::new(BigInteger256([
                    0x46bfb2ecbb75cd5c,
                    0xc76ae1ea7babab3b,
                    0x255b421b3292486b,
                    0x2fa34ee310c7abce,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x9178eb59ad64cabb,
                    0xef778765367b8c64,
                    0xa31f47c1a6871ff8,
                    0x28a51cc58bd8b12e,
                ])),
                Felt::new(BigInteger256([
                    0x06e9d8783f96c673,
                    0x004eb1bb548e127d,
                    0x0b57f8d28d03601f,
                    0x09ce082e7239f6b6,
                ])),
                Felt::new(BigInteger256([
                    0x27d484742dfac217,
                    0x90294bc406d38201,
                    0x79e090dd04e94ba5,
                    0x190f8452e4cd555c,
                ])),
                Felt::new(BigInteger256([
                    0x9948d28211a61a90,
                    0xfb48c7f2795be4df,
                    0xd27357ce509e7c1f,
                    0x20157ddd72f0cf30,
                ])),
                Felt::new(BigInteger256([
                    0x8a8e0a04cdac4e35,
                    0x930a9c6e7085d8d8,
                    0x4e0edd0f0388367f,
                    0x25be48d3e16e0ace,
                ])),
                Felt::new(BigInteger256([
                    0x1e25e67e44e50f45,
                    0x32269d9ffe081858,
                    0x83173ad51ebc23a9,
                    0x150ac92fc5b77381,
                ])),
                Felt::new(BigInteger256([
                    0xc33b4af201ddac84,
                    0x94047ab5f7f1385f,
                    0xb2fefb46c3b27798,
                    0x36560f4b2e139226,
                ])),
                Felt::new(BigInteger256([
                    0xd7866ca35fe71244,
                    0x4b4bde34f8b6e829,
                    0xbca33b33d94fc811,
                    0x17ca9f55f5a353a7,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xe633cadec3b9aa23,
                    0x583e3ab3278fe9b0,
                    0xcf15c9fcb1e39862,
                    0x1c1764ecf3085f2c,
                ])),
                Felt::new(BigInteger256([
                    0xa1781938a33895a4,
                    0xc1c50aecfe14372f,
                    0x2991aabe0c608cd0,
                    0x2428e9720863f5f3,
                ])),
                Felt::new(BigInteger256([
                    0x674f62423666e73f,
                    0xfa060c32cc362910,
                    0x548c409c7fff5789,
                    0x3977f149e18d0b33,
                ])),
                Felt::new(BigInteger256([
                    0x4ff1367c509ada41,
                    0x81b4a1280e07bd58,
                    0x3260bfb7298a5fd1,
                    0x2515c2b68ae8e1ef,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd865d6b0f7d4942c,
                    0xd1218f2a71ca3228,
                    0x463fa2825eb15dce,
                    0x3809a693c32da92a,
                ])),
                Felt::new(BigInteger256([
                    0xb036b106211eef25,
                    0xcff5fd8c595a764f,
                    0xa4be7c9de60dfee0,
                    0x3358c6af6bf04f59,
                ])),
                Felt::new(BigInteger256([
                    0xa5bfb1ee1e27a66f,
                    0xc2b66ef6b6929964,
                    0x75ab6b9f2db74a5e,
                    0x1d01f49c5d79a16a,
                ])),
                Felt::new(BigInteger256([
                    0x674959603df84a8e,
                    0x9039453ce52f8bf6,
                    0x755fa8bdf32f0d69,
                    0x290ff219f5f8da98,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0709498497fb576e,
                    0x6931b67f85bb8acd,
                    0x807fe6fbbcaf0dff,
                    0x2ee6c8e489437fd7,
                ])),
                Felt::new(BigInteger256([
                    0x3295ea8cb803fd9f,
                    0xf3367ee3c3c3df1e,
                    0x1de07324114f0ea7,
                    0x1bf83b93ca085472,
                ])),
                Felt::new(BigInteger256([
                    0x53c357d49a80864a,
                    0xcf5e8f8ed719bc73,
                    0x968a21dc60bbe098,
                    0x011433c1db7209eb,
                ])),
                Felt::new(BigInteger256([
                    0x27bec3c6dc82d4f8,
                    0x0b95758eb45b1ef4,
                    0x4628f35d0cf3557e,
                    0x1f714f2603b82f01,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xff6bd67be608724a,
                    0x822c2eac9f1d5705,
                    0x907f065db3ca6524,
                    0x00614cb177ba72b5,
                ])),
                Felt::new(BigInteger256([
                    0x6f67fdbb92469f8d,
                    0x68b99e8198b8c45c,
                    0x3475134047960cfd,
                    0x38c21631ff2bb0aa,
                ])),
                Felt::new(BigInteger256([
                    0xbe87654fe8fd5ba9,
                    0xf9d1751a3c0196aa,
                    0x5aca82ccf784ab99,
                    0x1330a628d182cfac,
                ])),
                Felt::new(BigInteger256([
                    0x40c5bc8675714039,
                    0x820c3d3439b0dd47,
                    0xe43581842d254cd4,
                    0x034a6ced6619c9cd,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd08b3973de0c488e,
                    0xd14e85c2fe64a88b,
                    0xbdbca95bd3cf2e13,
                    0x28cbd57bf4d180b7,
                ])),
                Felt::new(BigInteger256([
                    0x6a05fe1efaf3b83c,
                    0x144bab855d9ae34c,
                    0x79decc23b2df178b,
                    0x1243d015cb01a7fa,
                ])),
                Felt::new(BigInteger256([
                    0x2f084eef767a3a73,
                    0x85f85d34e6605445,
                    0x2c5040ae5bf95427,
                    0x12d2edb40c1f6c46,
                ])),
                Felt::new(BigInteger256([
                    0x402888099d36bab5,
                    0x6ccca4e6852c218d,
                    0x7bca8c914754aa1d,
                    0x02be4bfc73d96004,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x390000e185f908b7,
                    0x51bde52271bb329a,
                    0x102957d30469db61,
                    0x0da275eaaa9eb523,
                ])),
                Felt::new(BigInteger256([
                    0xa695040b987a971e,
                    0x89772f1be4a21ccb,
                    0x7400c549f2de3721,
                    0x2771e1b971387663,
                ])),
                Felt::new(BigInteger256([
                    0xe86d4eb2c9f5ba17,
                    0x8196e3a7f2a61931,
                    0x388f0bee92de3587,
                    0x388577fb2494e2fb,
                ])),
                Felt::new(BigInteger256([
                    0x9a424fd9004de1aa,
                    0x68f93efe478212c6,
                    0x93c9e806c2201b34,
                    0x319e8f429ef81a89,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc20eaae1d2c2360f,
                    0x07e70fc1060477f1,
                    0x533ce3b6ba267bd6,
                    0x3dfed7f86a921461,
                ])),
                Felt::new(BigInteger256([
                    0x8f06cf69dcd654ba,
                    0xbe753186ff5aafc0,
                    0x4cc0722ad015ecef,
                    0x1fb60157bb86bd8d,
                ])),
                Felt::new(BigInteger256([
                    0xf856f067e6c89e3a,
                    0x5ff67055cd6805cf,
                    0x1fb674d532be107f,
                    0x15353a342a15e7e3,
                ])),
                Felt::new(BigInteger256([
                    0xd658943470f5b6b8,
                    0x59ff31950b7fdfac,
                    0x3472ec44ec7a5e87,
                    0x16ec5d68e3e5f91c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xec77a730b83370ea,
                    0xf9460a467f152817,
                    0x012a7879e5a0e62c,
                    0x2139eb53c230b91e,
                ])),
                Felt::new(BigInteger256([
                    0x128f94e14975fd37,
                    0x765a745fff00208b,
                    0x66b5b5955e0688f1,
                    0x1da51238a0a70e86,
                ])),
                Felt::new(BigInteger256([
                    0x57e4594116fe6e95,
                    0xd21af61755d6c709,
                    0xaff60e2d15cdd69d,
                    0x3893e8b18fd3d4ed,
                ])),
                Felt::new(BigInteger256([
                    0x3e6adc6e7cc9948e,
                    0xd913b65e0cb37007,
                    0x1ca7488b3ab77afc,
                    0x11bcba4c90ebd950,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x001ec7ca239450a3,
                    0x4401be73b323d7cd,
                    0x0a68d5d23731205b,
                    0x2c3d812f27dd5d2d,
                ])),
                Felt::new(BigInteger256([
                    0x9eb900bbcbd6b870,
                    0x6b761e594243d526,
                    0x7987297195dbbd68,
                    0x1b994c7b15853201,
                ])),
                Felt::new(BigInteger256([
                    0x227d6503dc6e3711,
                    0x386349e3353faa14,
                    0x746f5c0cbcbc3d41,
                    0x1a9c1127d4324a63,
                ])),
                Felt::new(BigInteger256([
                    0x1f372a067dfe9f81,
                    0xd6868f671fbf61c3,
                    0x93a9c8186a6cd9cd,
                    0x06ff47a8f386111c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb07e4a26f016a278,
                    0x2aadb58bc499fdbf,
                    0xc700c23d3971c797,
                    0x38ce0c450e3257b3,
                ])),
                Felt::new(BigInteger256([
                    0xc4b6f12ceb772204,
                    0x4795518f922619e8,
                    0x4e47a4988196c231,
                    0x3b8490262da424a7,
                ])),
                Felt::new(BigInteger256([
                    0xd464a19b73189b56,
                    0xcc340ef9e35ebb16,
                    0x66ce49cf1fcc370f,
                    0x2508f2b011529cb7,
                ])),
                Felt::new(BigInteger256([
                    0xcaa73d4e878d5d42,
                    0x95b160f68c860c78,
                    0x75094538ee2f8970,
                    0x37e547a9b64dd889,
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
            vec![
                Felt::new(BigInteger256([
                    0x65954f92d761bfb8,
                    0xecd8a6f5df4adc20,
                    0x0de1e0a3a6587c24,
                    0x0592f4e43382828f,
                ])),
                Felt::new(BigInteger256([
                    0x37c7b65c971d1e6a,
                    0x47fc9344821b96b4,
                    0x2b449b807eb647f1,
                    0x06a2498b0862bdbd,
                ])),
                Felt::new(BigInteger256([
                    0x475da60a3fbfa519,
                    0x846bd2e8280babbf,
                    0x6d0e84420c61ebdb,
                    0x010e74d68e3484f5,
                ])),
                Felt::new(BigInteger256([
                    0xbfd2632effda5b9b,
                    0x5cc9320031f11d81,
                    0x02ad0d58ea9890be,
                    0x003502c7f0b05554,
                ])),
                Felt::new(BigInteger256([
                    0x2ccae29b6adfea8d,
                    0xf5be70af4078129c,
                    0x005065a516fde2f5,
                    0x3bd17502e42e364e,
                ])),
                Felt::new(BigInteger256([
                    0xdd3b3b995b8c85b7,
                    0x96617fad339ba193,
                    0x1230933eebc5da57,
                    0x21fe12526506d851,
                ])),
                Felt::new(BigInteger256([
                    0x1ca31b36978d1e40,
                    0x4bd336db864c69ea,
                    0xba15cccccba6cbaf,
                    0x236d6647bfe6010c,
                ])),
                Felt::new(BigInteger256([
                    0xf951b35767c0138e,
                    0xbd69bf0748cdc914,
                    0x6a4a535cac38aabc,
                    0x2f9297bdb2960987,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa7ec6d1fc9c0366e,
                    0x0e7661179d14745f,
                    0x7421a8b96331b5eb,
                    0x0f15dfa6465dcf88,
                ])),
                Felt::new(BigInteger256([
                    0x1950849af4a2f149,
                    0x179e0b231f0dfb0c,
                    0xb7aebedd993d0390,
                    0x37501538b698dced,
                ])),
                Felt::new(BigInteger256([
                    0x9669ede8e15ac492,
                    0x42c6f41c1904c86a,
                    0xeafe9ed245aeb0f3,
                    0x2e3cb85de287769a,
                ])),
                Felt::new(BigInteger256([
                    0x4b75bc966c9df7cf,
                    0xeba88e05819dcc7f,
                    0xc392ac783329998c,
                    0x0193182b5101ecb4,
                ])),
                Felt::new(BigInteger256([
                    0x4877c811e0f8b2ac,
                    0xff338ca74bde8593,
                    0x24c3e0c2dfa5b3bc,
                    0x0edb93d5ed71e1be,
                ])),
                Felt::new(BigInteger256([
                    0x153e128e566996ef,
                    0x7872bf0ee9c12c5c,
                    0xc08cf35668f7d31b,
                    0x176b02816ed7cf4c,
                ])),
                Felt::new(BigInteger256([
                    0x27561f9653b7e805,
                    0xbed6f20460d2c171,
                    0x8dcb2bd99706f767,
                    0x3e154359ab39060d,
                ])),
                Felt::new(BigInteger256([
                    0xdb27a578075f864a,
                    0xbd3d070f0a368487,
                    0xd97c1dae68d05ba6,
                    0x3c483f05f8dd07bf,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x9f9199eab096bb67,
                    0x668428a797613afb,
                    0x9a60babdf2d8f6e6,
                    0x1fb5c29b5fe47366,
                ])),
                Felt::new(BigInteger256([
                    0x3fc921e77e90581f,
                    0x7970d708e557e9bd,
                    0x86e5f8951a049bd2,
                    0x31a747bbb27b9e60,
                ])),
                Felt::new(BigInteger256([
                    0xf21b98b05b4fb99c,
                    0x71f056055686fdc8,
                    0x73dccb216e2142e7,
                    0x0e6b643c558f3ca5,
                ])),
                Felt::new(BigInteger256([
                    0x3120cd33c527f192,
                    0x41f5e67cc749a387,
                    0xcd0291f77cb4fe04,
                    0x0dde7fbe3afb4159,
                ])),
                Felt::new(BigInteger256([
                    0x1db0391d7e1c67f9,
                    0x2b3b8b081ab8673e,
                    0x1159aea9e4abe260,
                    0x3f94d5ecb7cf730d,
                ])),
                Felt::new(BigInteger256([
                    0x9ddb91c573acf6fe,
                    0x47f7f9e030ab6130,
                    0xd7e50536f906cc6f,
                    0x3d05b3a6f9a7068a,
                ])),
                Felt::new(BigInteger256([
                    0xc17f294c94c0767b,
                    0xf6b2ef35a89c2897,
                    0x5c82c78d3034757d,
                    0x209a5d0f31b177b6,
                ])),
                Felt::new(BigInteger256([
                    0x83cdcc5b83706de6,
                    0xcba18069da1ade76,
                    0xbb23bbb167a3ecf1,
                    0x19ccab896cd0674d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa86ca4f694d87be6,
                    0x57ef00825ad55041,
                    0x264490e1327b2fe5,
                    0x1570f07babc5fa1e,
                ])),
                Felt::new(BigInteger256([
                    0xa26447d950c3ed48,
                    0x23baa17acc0f6ba4,
                    0x01103fd1ef27f166,
                    0x2a2c6d57cf3b0782,
                ])),
                Felt::new(BigInteger256([
                    0x4c47239c3d251e88,
                    0xfe665d15d3d6bf3a,
                    0xd011111fdaa225a5,
                    0x088bd738b928b129,
                ])),
                Felt::new(BigInteger256([
                    0x65cc2c030abc32bd,
                    0x443eeceea73bd1de,
                    0xc1ba1b8707252c0c,
                    0x092e4d8f74eb17d9,
                ])),
                Felt::new(BigInteger256([
                    0x0996cf02ff223e56,
                    0x297d6836818332f9,
                    0x2a54816a90bf0021,
                    0x0ebada1f64477583,
                ])),
                Felt::new(BigInteger256([
                    0xceea6a0d24b23b7a,
                    0xd03ce62255f3f686,
                    0xc5eaf3f78d80fcec,
                    0x15b08d09ebfc68bc,
                ])),
                Felt::new(BigInteger256([
                    0x4cfe70c6c9411ba6,
                    0x7ef39b2b2459a494,
                    0x57b53ae993b01ce8,
                    0x1c796b9662df0067,
                ])),
                Felt::new(BigInteger256([
                    0x11883d8d0153586d,
                    0xbfdab2b13f172a24,
                    0xc172348853b956e1,
                    0x03e8aeeaa25b6f8c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x6114d057a67c5930,
                    0xfade0064b2aab47a,
                    0x3e310f7e8d33c8e5,
                    0x3546fb240ecbd172,
                ])),
                Felt::new(BigInteger256([
                    0x4271b8f55559cf00,
                    0x1055b8e103ef06bd,
                    0x51f7e3fad03dd98c,
                    0x2bc4d34abc9bad75,
                ])),
                Felt::new(BigInteger256([
                    0x6305afa915eaad3b,
                    0x1fd3be3a93de62b1,
                    0x0d053853a45d37f5,
                    0x092fca895b756c79,
                ])),
                Felt::new(BigInteger256([
                    0x8ecdc40dacbaec3c,
                    0xa8f921ae14691a88,
                    0xd30b632f2a9a9ba1,
                    0x3d03891abd9b4707,
                ])),
                Felt::new(BigInteger256([
                    0x4edda04daa32f540,
                    0xfc231944f9462ddd,
                    0x8a5c6ec9f43f370a,
                    0x03381f6c28c9e63f,
                ])),
                Felt::new(BigInteger256([
                    0x685c85456b2333d2,
                    0x31e16b229e95f906,
                    0x8f00f056c66988e7,
                    0x1612d5f2c139527b,
                ])),
                Felt::new(BigInteger256([
                    0x23a85f7b222363fd,
                    0x2a6f164a89a89378,
                    0x0086c3360d479adf,
                    0x318498ea2675fbac,
                ])),
                Felt::new(BigInteger256([
                    0x4afda0b3cea6d952,
                    0x4324dc543fcb1857,
                    0x04a5be14d06c02e3,
                    0x3cbc673a4cceb85e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x84287c3edce28f12,
                    0x05fdd997aba82cdf,
                    0x11327227973de99a,
                    0x0d3eadb2604f5c30,
                ])),
                Felt::new(BigInteger256([
                    0x917a1d2995a87542,
                    0x744819d4fbca174d,
                    0x1ed0d6e4ad9610ba,
                    0x2c41a3163dc25708,
                ])),
                Felt::new(BigInteger256([
                    0xd37e3e99b27c8ca4,
                    0x48687f09cb62a0d7,
                    0x3b9b34330b2658c0,
                    0x0f3321f153346337,
                ])),
                Felt::new(BigInteger256([
                    0xde1d3a84274a9f7c,
                    0x05ec41cb61cd8fd3,
                    0x70f138c2e27ba802,
                    0x26ac30ff4b4e9dcf,
                ])),
                Felt::new(BigInteger256([
                    0x0ccb347c0dce391e,
                    0x7c3a0ee82f843402,
                    0xb566472679b44f9f,
                    0x196a8fb50ab0c389,
                ])),
                Felt::new(BigInteger256([
                    0x6b0d498ff635511c,
                    0x57e136ea00bde584,
                    0x1e8935a24935bc13,
                    0x39b312a09346071e,
                ])),
                Felt::new(BigInteger256([
                    0x9284211964958d9f,
                    0x4229e9aaed9a9df0,
                    0xb742099e2c78078a,
                    0x1c086a07aad04f2b,
                ])),
                Felt::new(BigInteger256([
                    0x54335f97edab5922,
                    0xec58b3187a45e0ef,
                    0x64332116b18c13a4,
                    0x0e85d21884c3f697,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xc13c41fffa209161,
                    0x2ffdade9ea3169e3,
                    0x23a20a9931e2efec,
                    0x158f5636d4956a60,
                ])),
                Felt::new(BigInteger256([
                    0x65226493f3d36fe4,
                    0x2133131902874baa,
                    0x5bf26a7535eaeca2,
                    0x093eac28934cd7e2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf1de9d7e15fc3a9a,
                    0x719165251ec822af,
                    0xbbeb0e218c68a82d,
                    0x150b9b3020a74a94,
                ])),
                Felt::new(BigInteger256([
                    0x8b391f455f1739b2,
                    0x3de8a9cd34f55968,
                    0x1a1e255bd93d0c4a,
                    0x1c68b8c961e929f2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5acca159327bddb8,
                    0x3890460e5cd54740,
                    0x170a08d81d6aee98,
                    0x2ffafca664b589c3,
                ])),
                Felt::new(BigInteger256([
                    0x5a54ae539486d297,
                    0xfecbf472781efe12,
                    0x640966811e426425,
                    0x3b698ab9cdc08373,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbdf33bcbcf05cdf3,
                    0x7bfda3c6db1eedb0,
                    0xeb49892aab4f10be,
                    0x1391f2da493d4261,
                ])),
                Felt::new(BigInteger256([
                    0xb02dba4207b7dfc6,
                    0xeac5dbb5d269a1a3,
                    0x18aa94c474bb59d1,
                    0x3c0c831f65457a78,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3de0d4982c21c72f,
                    0xa3684312749120e1,
                    0x2df68695ac16d57b,
                    0x0757c8349bfaae2d,
                ])),
                Felt::new(BigInteger256([
                    0x653c835f91ec8442,
                    0xd1676e87dcb827a2,
                    0xfc1a274685d1fcb1,
                    0x3993f1d00f2680c3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x542497c9611df6fe,
                    0xad9b8914795d20f7,
                    0x275203530acfac83,
                    0x3a6a6bd0374e954d,
                ])),
                Felt::new(BigInteger256([
                    0x6e12935286691954,
                    0x437750d0dd782ba4,
                    0xa962da8fa243e8ad,
                    0x07d9a3834c746e04,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa1147060e2abb0eb,
                    0xeac55876d96bfa18,
                    0xa4a79a3c697e0ce7,
                    0x0e4b7d1b38a1c23e,
                ])),
                Felt::new(BigInteger256([
                    0xee8d0d688581be79,
                    0xb45917a45cfa425f,
                    0xb581b025c9690646,
                    0x376f93a66600ad3a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x17575af28ea45c4a,
                    0xddeaeb8210ae2b44,
                    0x7c06d0bd88693160,
                    0x37e22ecb2b9d2350,
                ])),
                Felt::new(BigInteger256([
                    0x70dee65d2b67802c,
                    0x3f0325531a644b20,
                    0x22e120c1c0e77179,
                    0x05647969c0c5bb22,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x93fd1d84913ddd89,
                    0x522e6990149bdf6a,
                    0x7ec7418c1e26ef0f,
                    0x3b856dc3a460d824,
                ])),
                Felt::new(BigInteger256([
                    0xc49dcbe1dca5534e,
                    0x2a5cb1a9df050d6b,
                    0x421e4b45f49951cd,
                    0x3681dbaf06d0b1c8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x210eec77d2d870c0,
                    0x8616748bce753032,
                    0x0136465d8bed3c2e,
                    0x24255b9ba5e158cc,
                ])),
                Felt::new(BigInteger256([
                    0x5290bd56e6c0eb85,
                    0x389efe85f3309f56,
                    0xe60ae07728ecf26d,
                    0x32fccd97aeb14e6e,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
