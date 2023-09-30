//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiVesta_2_1, Jive, Sponge};
use super::{DIGEST_SIZE, STATE_WIDTH};
use crate::traits::Anemoi;
use ark_ff::PrimeField;

use super::Felt;
use super::{One, Zero};

impl Sponge<Felt> for AnemoiVesta_2_1 {
    type Digest = AnemoiDigest;

    fn hash(bytes: &[u8]) -> Self::Digest {
        // Compute the number of field elements required to represent this
        // sequence of bytes.
        let num_elements = if bytes.len() % 31 == 0 {
            bytes.len() / 31
        } else {
            bytes.len() / 31 + 1
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 31-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut buf = [0u8; 32];
        for (i, chunk) in bytes.chunks(31).enumerate() {
            if i < num_elements - 1 {
                buf[0..31].copy_from_slice(chunk);
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
            state[0] += Felt::from_le_bytes_mod_order(&buf[..]);
            AnemoiVesta_2_1::permutation(&mut state);
        }
        state[STATE_WIDTH - 1] += Felt::one();

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn hash_field(elems: &[Felt]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        for &element in elems.iter() {
            state[0] += element;
            AnemoiVesta_2_1::permutation(&mut state);
        }

        state[STATE_WIDTH - 1] += Felt::one();

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn merge(digests: &[Self::Digest; 2]) -> Self::Digest {
        // We use internally the Jive compression method, as compressing the digests
        // through the Sponge construction would require two internal permutation calls.
        let result = Self::compress(&Self::Digest::digests_to_elements(digests));
        Self::Digest::new(result.try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiVesta_2_1 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.to_vec();
        AnemoiVesta_2_1::permutation(&mut state);

        vec![state[0] + state[1] + elems[0] + elems[1]]
    }

    fn compress_k(elems: &[Felt], k: usize) -> Vec<Felt> {
        // This instantiation only supports Jive-2 compression mode.
        assert!(k == 2);

        Self::compress(elems)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::super::MontFp;
    use super::*;
    use ark_ff::BigInteger;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![MontFp!(
                "23666212284893956228650649523052266029381274089955769026913903282248093405315"
            )],
            vec![
                MontFp!(
                    "3501078521102541220367678465136405970430548314172589887937956121704171293165"
                ),
                MontFp!(
                    "21515037255667073903154594949079633786175296923471185077272491054680582398738"
                ),
            ],
            vec![
                MontFp!(
                    "26245066117244349358562045231373743633930400064101767628209065741336089170284"
                ),
                MontFp!(
                    "14006547378986078708925107537283765026889558540769120987741627559095518263486"
                ),
                MontFp!(
                    "16319987390108120672751685067412732420038413171290841893567961226113025211588"
                ),
            ],
            vec![
                MontFp!(
                    "24641144407733305733584202825717792208451054286148307900846544116227059877590"
                ),
                MontFp!(
                    "4228524054177173784724834792175558882315579450456758510161598683145463027415"
                ),
                MontFp!(
                    "25306022459096007145609287595234936859365285245289971425284966567494172328321"
                ),
                MontFp!(
                    "8175349843958286223153744941633339765213483300995083554411569483944287197510"
                ),
            ],
            vec![
                MontFp!(
                    "8740657817957640589599155262884611618719235709713726089073814688077491547535"
                ),
                MontFp!(
                    "16595167152966320362582352064315089504560491581356149685807692878013337924241"
                ),
                MontFp!(
                    "14064094898173722222310755363414374056362918969089607341720950961732912315011"
                ),
                MontFp!(
                    "800124860363430545351049000046721525489884728528871381794803143175020182052"
                ),
                MontFp!(
                    "17623916170046312159115432199378103392440393591499193444414499557262400227578"
                ),
            ],
            vec![
                MontFp!(
                    "8559181616607833234020874084936434756111691716976821157781711259378342247349"
                ),
                MontFp!(
                    "3712032813022748960353736644218860630753855350311785475557015515761931707021"
                ),
                MontFp!(
                    "18244586813745203405426970765128722714527216813367626489683188870729460745061"
                ),
                MontFp!(
                    "4625304081143527908085723806207105973411010421613082699340143544709821862785"
                ),
                MontFp!(
                    "1579643740689782728856536923239854836285170264865496072329853825108973138439"
                ),
                MontFp!(
                    "26615051575295251925147288423474991536248319820477111544037388634970756375021"
                ),
            ],
        ];

        let output_data = [
            [MontFp!(
                "3675563954394682696996536055507407419428734133672283731223142852773229935377"
            )],
            [MontFp!(
                "2229117904311058771450486580798876451943541707571959402671744982352730393011"
            )],
            [MontFp!(
                "7188444258209023690460633892327566210376620195026849798399800429667062149490"
            )],
            [MontFp!(
                "9466600354778764410480247555459461306567613323441414865846416198947210643152"
            )],
            [MontFp!(
                "22306538566530453062617248694580445489296260848593489687893794609592376195218"
            )],
            [MontFp!(
                "23212306122676917261335277337636005824963471853164563343020044774792071907230"
            )],
            [MontFp!(
                "14610922761737438776547487420770558826022222657127880833279093942376034354577"
            )],
            [MontFp!(
                "4117277901691807903515683082396918651140029386316771349589349676343065811802"
            )],
            [MontFp!(
                "25932136303950551788718747693050119626519481083317358740958528616374287701013"
            )],
            [MontFp!(
                "13391622674438414062657793529140174149205584287960651726917252462804048527287"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiVesta_2_1::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_hash_bytes() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
        ];

        let output_data = [
            [MontFp!(
                "3675563954394682696996536055507407419428734133672283731223142852773229935377"
            )],
            [MontFp!(
                "2229117904311058771450486580798876451943541707571959402671744982352730393011"
            )],
            [MontFp!(
                "7188444258209023690460633892327566210376620195026849798399800429667062149490"
            )],
            [MontFp!(
                "9466600354778764410480247555459461306567613323441414865846416198947210643152"
            )],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 62];
            bytes[0..31].copy_from_slice(&input[0].into_bigint().to_bytes_le()[0..31]);
            bytes[31..62].copy_from_slice(&input[1].into_bigint().to_bytes_le()[0..31]);

            assert_eq!(expected, AnemoiVesta_2_1::hash(&bytes).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
        ];

        let output_data = [
            [MontFp!(
                "25021493149962073135159779742595331542889244616612316327827500113201517573552"
            )],
            [MontFp!(
                "1799222279508491238955156019299185816766170120519060796492407909371488003482"
            )],
            [MontFp!(
                "3312133520551670415496942812397107034820036209574409986611422513836613460957"
            )],
            [MontFp!(
                "3814795182979631662421649092648209217376074705260062533460857166180948182756"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiVesta_2_1::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiVesta_2_1::compress_k(input, 2));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiVesta_2_1::merge(&[
                    AnemoiDigest::new([input[0]]),
                    AnemoiDigest::new([input[1]])
                ])
                .to_elements()
            );
        }
    }
}
