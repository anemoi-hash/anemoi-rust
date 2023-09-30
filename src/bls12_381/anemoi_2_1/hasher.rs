//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiBls12_381_2_1, Jive, Sponge};
use super::{DIGEST_SIZE, STATE_WIDTH};
use crate::traits::Anemoi;
use ark_ff::PrimeField;

use super::Felt;
use super::{One, Zero};

impl Sponge<Felt> for AnemoiBls12_381_2_1 {
    type Digest = AnemoiDigest;

    fn hash(bytes: &[u8]) -> Self::Digest {
        // Compute the number of field elements required to represent this
        // sequence of bytes.
        let num_elements = if bytes.len() % 47 == 0 {
            bytes.len() / 47
        } else {
            bytes.len() / 47 + 1
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 47-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut buf = [0u8; 48];
        for (i, chunk) in bytes.chunks(47).enumerate() {
            if i < num_elements - 1 {
                buf[0..47].copy_from_slice(chunk);
            } else {
                // If we are dealing with the last chunk, it may be smaller than 47 bytes long, so
                // we need to handle it slightly differently. We also append a byte set to 1 to the
                // end of the string if needed. This pads the string in such a way that adding
                // trailing zeros results in a different hash.
                let chunk_len = chunk.len();
                buf = [0u8; 48];
                buf[..chunk_len].copy_from_slice(chunk);
                // [Different to paper]: We pad the last chunk with 1 to prevent length extension attack.
                if chunk_len < 47 {
                    buf[chunk_len] = 1;
                }
            }

            // Convert the bytes into a field element and absorb it into the rate portion of the
            // state. An Anemoi permutation is applied to the internal state if all the the rate
            // registers have been filled with additional values. We then reset the insertion index.
            state[0] += Felt::from_le_bytes_mod_order(&buf[..]);
            AnemoiBls12_381_2_1::permutation(&mut state);
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
            AnemoiBls12_381_2_1::permutation(&mut state);
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

impl Jive<Felt> for AnemoiBls12_381_2_1 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.to_vec();
        AnemoiBls12_381_2_1::permutation(&mut state);

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
                "3830311086848849888236431976110302247116143110896873177609502790121170427286440444824651043474707388926317147242998"
            )],
            vec![
                MontFp!(
                    "1724675944989629274545203806912510407614247173442012446934556163001278695817727182442175312153616433074255922703832"
                ),
                MontFp!("3105146794553518502095497824949882888889680326951826917122755101075932697139414494188338481262964404838634130364755"
                ),
            ],
            vec![
                MontFp!(
                    "350492308445544793552494864949788846839916342308595111620239094560542916422774935668417321560437725412255000974728"
                ),
                MontFp!("1725309941267812667675900676635576215123627147507130530667608213130494035120812807635577850015904982632428754346146"
                ),
                MontFp!("2312947547030282178092241201149116938180736364300301327043058201125277110757024339134164864808561362997575042255525"
                ),
            ],
            vec![
                MontFp!(
                    "3636407125870859276435295141718539694027483654558256793971395008553891302813361556095695045118168186186809341860437"
                ),
                MontFp!("1394179633736664718629986299474681395660571356666052573125212216562117078458807048111125315791237011833872591487687"
                ),
                MontFp!("3000192289890877131066307906160796735894710747067802082952862583101732077656564864996804222836453117161416409451060"
                ),
                MontFp!("2064900497271051602050546834876568795558276926185454314976506949213452504584265729843506251928443518977086936047522"
                ),
            ],
            vec![
                MontFp!(
                    "605052123655532251743639038404178558281286158572543759898583863142120387035833720039011317858867488262299677045948"
                ),
                MontFp!("1378646499724679273279330327531363802830770406492481700661528470796382880277680062911887140309185106460282485023138"
                ),
                MontFp!("1632789037231732221874565879516499749343606218784801731743384108930268889394243187732723538359946499168567705027090"
                ),
                MontFp!("329592553648563403858723137432007556870910416840415359696339423598921411929428890497411782460237276991297868386396"
                ),
                MontFp!("1195876680499641105595204824797249899369568299490932910431218696161934043072334617503409121187788565077168019563184"
                ),
            ],
            vec![
                MontFp!(
                    "3075190921122359950664934703601648556085420058671951436958011401587534690551115717377111396274988564198237304353666"
                ),
                MontFp!("3419326986335584916312489712000516327290696677572970766585889391616104705346363375577222750937246008867925423482798"
                ),
                MontFp!("1378948475731859253682901417906582577745453964942678637296873800370897690266489687583065871730347771936433732173956"
                ),
                MontFp!("434819213209834856425262399894697513386422854458119916389578950785921826099147747092122973654613949117993887158169"
                ),
                MontFp!("2959811739045932941263291579672134993493582127052200852956262762274535462648144476602886585403619686645783651792211"
                ),
                MontFp!("2002518377401644274294207739192197222547494224748071260569678901651938265396145032803394648935423898359933682789880"),],];

        let output_data = [
[MontFp!("3044496648238508402971476696005782461598801178368833793411335711455208110605169771916004504371572228154710324741161"),],
[MontFp!("2908793991367076052361840957762986139066526738758775834650765236473336488638852269315020736340802721910347134703704"),],
[MontFp!("2944094137414315537156609151358641099704493279307534425028271048135173488544687230539814152428883365064015829820463"),],
[MontFp!("2868958625495483458583620051234160973380283203362973188133402396622476242457329234265515602902085044122040631694899"),],[MontFp!("1866033995575078117872271638683543620649307607095016362529898258120178575148148641617480439242769826709240970576077"),],[MontFp!("718258795897765944758177208463485114194620088839955789078982618553526798718502557788643069426154127435845854471100"),],[MontFp!("2286698192191934508507293243065494603958433091556657060465591121808349675884095770128733564548760243177375970842740"),],[MontFp!("1873646942936659856912960773765807616353120774536672458029128234873732160402675620321975596948087041724675387375220"),],[MontFp!("214854368678843434181525108886989867124405520390579022106184552736389572975439870185673104599101705970513292871653"),],[MontFp!("3625784097092860341146015211036247185380543997103305980159712597275345478788335407338478610702273655584618860197247"),],];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiBls12_381_2_1::hash_field(input).to_elements()
            );
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
            [MontFp!("3044496648238508402971476696005782461598801178368833793411335711455208110605169771916004504371572228154710324741161"),],
[MontFp!("2908793991367076052361840957762986139066526738758775834650765236473336488638852269315020736340802721910347134703704"),],
[MontFp!("2944094137414315537156609151358641099704493279307534425028271048135173488544687230539814152428883365064015829820463"),],
[MontFp!("2868958625495483458583620051234160973380283203362973188133402396622476242457329234265515602902085044122040631694899"),],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 94];
            bytes[0..47].copy_from_slice(&input[0].into_bigint().to_bytes_le()[0..47]);
            bytes[47..94].copy_from_slice(&input[1].into_bigint().to_bytes_le()[0..47]);

            assert_eq!(expected, AnemoiBls12_381_2_1::hash(&bytes).to_elements());
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
[MontFp!("2403587970030648969605860549704915622710356578540224073435388894943352899101850938559087153461043348245106307692021"),],[MontFp!("3338092111784861946006125665732871699018939986325206562704758912134694966085752607725062142754026064165834984422908"),],[MontFp!("1985125909818301697005337958894654664774949870969130453343225221846235685271877058212539274671736851151869436601343"),],[MontFp!("3251723048147411152374473396236461385859897497022721086334583719384662681448082382905053326928560538637353089836071"),],];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBls12_381_2_1::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBls12_381_2_1::compress_k(input, 2));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiBls12_381_2_1::merge(&[
                    AnemoiDigest::new([input[0]]),
                    AnemoiDigest::new([input[1]])
                ])
                .to_elements()
            );
        }
    }
}
