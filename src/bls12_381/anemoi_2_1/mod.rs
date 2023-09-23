//! Implementation of the Anemoi permutation

use super::{sbox, Felt, MontFp};
use crate::{Anemoi, Jive, Sponge};
use ark_ff::{One, Zero};
/// Digest for Anemoi
mod digest;
/// Sponge for Anemoi
mod hasher;
/// Round constants for Anemoi
mod round_constants;

pub use digest::AnemoiDigest;

// ANEMOI CONSTANTS
// ================================================================================================

/// Function state is set to 2 field elements or 96 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 2;
/// 1 element of the state is reserved for rate.
pub const RATE_WIDTH: usize = 1;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 1;

/// One element (48-bytes) is returned as digest.
pub const DIGEST_SIZE: usize = RATE_WIDTH;

/// The number of rounds is set to 21 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 21;

// ANEMOI INSTANTIATION
// ================================================================================================

/// An Anemoi instantiation over BLS12_381 basefield with 1 column and rate 1.
#[derive(Debug, Clone)]
pub struct AnemoiBls12_381_2_1;

impl<'a> Anemoi<'a, Felt> for AnemoiBls12_381_2_1 {
    const NUM_COLUMNS: usize = NUM_COLUMNS;
    const NUM_ROUNDS: usize = NUM_HASH_ROUNDS;

    const WIDTH: usize = STATE_WIDTH;
    const RATE: usize = RATE_WIDTH;
    const OUTPUT_SIZE: usize = DIGEST_SIZE;

    const ARK_C: &'a [Felt] = &round_constants::C;
    const ARK_D: &'a [Felt] = &round_constants::D;

    const GROUP_GENERATOR: u32 = sbox::BETA;

    const ALPHA: u32 = sbox::ALPHA;
    const INV_ALPHA: Felt = sbox::INV_ALPHA;
    const BETA: u32 = sbox::BETA;
    const DELTA: Felt = sbox::DELTA;

    fn exp_by_inv_alpha(x: Felt) -> Felt {
        sbox::exp_by_inv_alpha(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sbox() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let mut input = [
            [Felt::zero(), Felt::zero()],
            [Felt::one(), Felt::one()],
            [Felt::zero(), Felt::one()],
            [Felt::one(), Felt::zero()],
            [MontFp!("2639505791130699847138581246185885055990948498489342305074770502258851288122761936413995271029951303782268594261940"
                ),
                MontFp!("2453803074341438677978248529076694173935112020474275107054519301539345953282344217007938580499167344755279671258477"),],[MontFp!("3826343494652146910587478037317662252994425647339827885932244558441507927524070523862223596722047642153049519958502"
                ),
                MontFp!("2213285664123851154164399517476276086297667175750394056833936829656263514994210546846041380414442943349942556131167"),],[MontFp!("1784235328212465922753432632718728940066227845167015006250965466134563711760247547998512753761878692738260037415320"
                ),
                MontFp!("39867492833405729841610794572020155220857746021404149003111105376414058511034326417887712183989738999685433791746"),],[MontFp!("1276805555513368654115849300586155645322547258070524299376715810359761599586232204047255985442457747688287778827514"
                ),
                MontFp!("2982607816330514123837188598555171983314899149704293255595313129175663904381898191258274873537791043974472985987138"),],[MontFp!("222277026251992776078878530826241946946986457602164080890346233257384167341250003945691162194269902668781766410462"
                ),
                MontFp!("2057200503249324857095407481415403298530252968820526499874388237895978705973147153711599050530453697929352884509172"),],[MontFp!("3722460117630412498731569280173727988220205878678823487737905245511347157625796082399677207483561644332555197168768"
                ),
                MontFp!("3882220687494333104644860313120672938730758190390321402327320632977900915881572611838696400124643490862957806853209"),],];

        let output = [
[MontFp!("2001204777610833696708894912867952078278441409969503942666029068062015825245418932221343814564507832018947136279894"),Felt::zero(),],[MontFp!("2001204777610833696708894912867952078278441409969503942666029068062015825245418932221343814564507832018947136279901"
                ),
                MontFp!("2"),],[MontFp!("3409024350324770932167116073246649688369598710716962856079695054623192887746906059977244451934321045232801607693752"
                ),
                MontFp!("1329263574752159191211793849573133266716936406142823755742645842982909514268936466892710969217855617281191220461747"),],[MontFp!("2001204777610833696708894912867952078278441409969503942666029068062015825245418932221343814564507832018947136279897"
                ),
                MontFp!("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559786"),],[MontFp!("2465574469952643145729746234477924421882674096882370412379683948302410643217957469321380693187563210193334655335847"
                ),
                MontFp!("2489152704958805531135773890840855334442445700083109664226974417610635874611932888877037573244368603578580414041799"),],[MontFp!("408652350564110577322969651688855407657559708877495483663280689285273237419309108571745248532063147969920029280822"
                ),
                MontFp!("1724486001880100049532256073750925567031848981057838120554348912902128069838138958515666882468688879658762982998568"),],[MontFp!("1765091336843926043630366340550371749829789993174128990666219612137983469744999990928422749620039381212355138385554"
                ),
                MontFp!("2480980614979128307054616596744604853000819434901463816868701456456203229459184563106027550124998454431518049919350"),],[MontFp!("2950346306930083665848529180886105408422777543098716242160201581561457223880734312067977330112099260792507514986360"
                ),
                MontFp!("2087374146453732791551455846913217956996861872358134514729576187731786238798141582308800938194775173567582922075705"),],[MontFp!("3402125800877092047974772179557145155027105993715549752961881793057712744390710655211608258897689691916462045687307"
                ),
                MontFp!("3403977659585714744033605508019495483596883599235937591469491142673507602627003480280825557962552759214007258462562"),],[MontFp!("3595429903081617974839182398366971282489553094052996224478569522592700873909643276467558825831974433870937620786666"
                ),
                MontFp!("2977853375622078966481740257910454397845244197779424503641005093520724697563286535134594629934135273187959765909140"),],];

        for i in input.iter_mut() {
            AnemoiBls12_381_2_1::sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
