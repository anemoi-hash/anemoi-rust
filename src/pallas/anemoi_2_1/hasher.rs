//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiPallas_2_1, Jive, Sponge};
use super::{DIGEST_SIZE, STATE_WIDTH};
use crate::traits::Anemoi;
use ark_ff::PrimeField;

use super::Felt;
use super::{One, Zero};

impl Sponge<Felt> for AnemoiPallas_2_1 {
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
            AnemoiPallas_2_1::permutation(&mut state);
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
            AnemoiPallas_2_1::permutation(&mut state);
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

impl Jive<Felt> for AnemoiPallas_2_1 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.to_vec();
        AnemoiPallas_2_1::permutation(&mut state);

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
                "8786344908157484467702144443147728582570609298051356647558148413464066108428"
            )],
            vec![
                MontFp!(
                    "3703611713748744165229098306080671845129048809115003457470291999105546486325"
                ),
                MontFp!(
                    "24153765309843050828623012216779077908319465245142980318434291969963354926334"
                ),
            ],
            vec![
                MontFp!(
                    "358913411029015325318138375435731803790992617178880485013196553608252539481"
                ),
                MontFp!(
                    "8230022665545053873997498884151270366890470487501907178234159557579737481432"
                ),
                MontFp!(
                    "2047386461740345715599555656917867466628251271969723740633193819624724688488"
                ),
            ],
            vec![
                MontFp!(
                    "26036341263213017331904003562532912502792005336457047110314507104201169419876"
                ),
                MontFp!(
                    "24062649992427728021819494567227387940731985117940055830958150360565836942248"
                ),
                MontFp!(
                    "8921365708708167109938294177011741430366329121282058520981996333836959700069"
                ),
                MontFp!(
                    "7195157583406215915072568967368134796399291042697277049061440998502574921737"
                ),
            ],
            vec![
                MontFp!(
                    "21356927064953552060503431555205407909741461873987532728641908180721700084840"
                ),
                MontFp!(
                    "7966758083609056599281293435577685235629680655349403837025696737240329722986"
                ),
                MontFp!(
                    "24135357023371509759002187273182309193982925096610998799260755998614247349066"
                ),
                MontFp!(
                    "10588941192463807739722749656640339239064042571914912104463662343538225353648"
                ),
                MontFp!(
                    "6839118740961792995837243286714037465343480861961456510297280599484025888117"
                ),
            ],
            vec![
                MontFp!(
                    "24183859570943675199564758425190682354367562032419636289128612513970763668072"
                ),
                MontFp!(
                    "18915354595390571159250786067138004725796825626596000701464917307766907388790"
                ),
                MontFp!(
                    "23236759124405113083148363509694740555047269126008151654831625763404594146823"
                ),
                MontFp!(
                    "8820782127262986192829346492854126768655317388934818603090790012122262224323"
                ),
                MontFp!(
                    "15314973457919135327168897691453063865004877213365317319046997450929013432725"
                ),
                MontFp!(
                    "26717381452276117819889657178273802186002762982479594599901892137785841519703"
                ),
            ],
        ];

        let output_data = [
            [MontFp!(
                "27921818220805746051987267737218857588847751174833938699888112568779408580714"
            )],
            [MontFp!(
                "10003429244477285855747230809015634417193821429428555712832755192897533334106"
            )],
            [MontFp!(
                "13290391632215095401717364136096553912180237160090779341925788216826661533240"
            )],
            [MontFp!(
                "91993706971719007054866387319808123857126926105539303153205824718341931836"
            )],
            [MontFp!(
                "4542202078280494022684418795562703153571828559807751715843580237564183190018"
            )],
            [MontFp!(
                "6189149292975300181581176863094298311283007276406715688822500697364177384640"
            )],
            [MontFp!(
                "20772154299187801558781683699482888401245070406146164128826035801084072271363"
            )],
            [MontFp!(
                "12766169934011665030732729437530902277038872474141285443644059071867016845440"
            )],
            [MontFp!(
                "7247086375400245676536480550203037779342035687559407087041004416219288636815"
            )],
            [MontFp!(
                "26406354915723717036607769016580923969246177561932524143753201372077323307043"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiPallas_2_1::hash_field(input).to_elements());
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
                "27921818220805746051987267737218857588847751174833938699888112568779408580714"
            )],
            [MontFp!(
                "10003429244477285855747230809015634417193821429428555712832755192897533334106"
            )],
            [MontFp!(
                "13290391632215095401717364136096553912180237160090779341925788216826661533240"
            )],
            [MontFp!(
                "91993706971719007054866387319808123857126926105539303153205824718341931836"
            )],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 62];
            bytes[0..31].copy_from_slice(&input[0].into_bigint().to_bytes_le()[0..31]);
            bytes[31..62].copy_from_slice(&input[1].into_bigint().to_bytes_le()[0..31]);

            assert_eq!(expected, AnemoiPallas_2_1::hash(&bytes).to_elements());
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
                "25339990266271823042514121456004268429320463644759496063425849229386076340157"
            )],
            [MontFp!(
                "13925721908743741981737180373217062710291818499780666961262790653502874125673"
            )],
            [MontFp!(
                "8161124099286768706096187391267581535071065397484800021294905474746837811202"
            )],
            [MontFp!(
                "22464570152859425264804262779232945996568499466977613824889819037772248156041"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiPallas_2_1::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiPallas_2_1::compress_k(input, 2));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiPallas_2_1::merge(&[
                    AnemoiDigest::new([input[0]]),
                    AnemoiDigest::new([input[1]])
                ])
                .to_elements()
            );
        }
    }
}
