??$
?%?%
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
=
Greater
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedBincount

splits	
values"Tidx
size"Tidx
weights"T
output"T"
Tidxtype:
2	"
Ttype:
2	"
binary_outputbool( 
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
?
StringNGrams
data
data_splits"Tsplits

ngrams
ngrams_splits"Tsplits"
	separatorstring"
ngram_widths	list(int)("
left_padstring"
	right_padstring"
	pad_widthint" 
preserve_short_sequencesbool"
Tsplitstype0	:
2	
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.12unknown8??#
?
RMSprop/dense_2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_2/bias/rms
?
,RMSprop/dense_2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_2/bias/rms*
_output_shapes
:*
dtype0
?
RMSprop/dense_2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *+
shared_nameRMSprop/dense_2/kernel/rms
?
.RMSprop/dense_2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_2/kernel/rms*
_output_shapes

: *
dtype0
?
RMSprop/dense_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameRMSprop/dense_1/bias/rms
?
,RMSprop/dense_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/bias/rms*
_output_shapes
: *
dtype0
?
RMSprop/dense_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *+
shared_nameRMSprop/dense_1/kernel/rms
?
.RMSprop/dense_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/kernel/rms*
_output_shapes

:@ *
dtype0
?
RMSprop/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameRMSprop/dense/bias/rms
}
*RMSprop/dense/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/bias/rms*
_output_shapes
:@*
dtype0
?
RMSprop/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*)
shared_nameRMSprop/dense/kernel/rms
?
,RMSprop/dense/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/kernel/rms* 
_output_shapes
:
??@*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
|
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_7*
value_dtype0	
l

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1317*
value_dtype0	
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

: *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
: *
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@ *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??@*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
Q
Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 R	????????
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
??
Const_4Const*
_output_shapes

:??*
dtype0*Ƨ
value??B????BtheBofB2Bof theB3BeditionBworldBiiBepisodeB4BandBaBdeadBgameBstarBwarsBinBwarBforBtoBsuperBdragonBdarkBheroesBand theBiiiBnbaBbattleB
collectionBfinalBevilBvsBsoccerBultimateBnflBcallB5BwarriorsB	star warsBproBlegendBhdBnewBracingBfifaBtourBcall ofBfantasyBlegoB
chroniclesBmanB1BforceBtomBmarioBtom clancysBclancysBlostBageBblackBfinal fantasyB	adventureBresident evilBresidentBcityBlegendsBquestB	legend ofBtimeBof dutyBninjaBkingBdutyBnhlBseriesBspaceBshadowBstreetBredB
remasteredBlifeBriseBisBchampionshipBsonicBoneBtwoBspeedB	assassinsB	evolutionBonlineBmegaBcreedBballB
adventuresBbaseballBrise ofBassassins creedBmonsterB
madden nflBmaddenBheroBbloodBsimsBnightBneed forBneedBmetalBivB	for speedBwalking deadBwalkingBthe walkingBbatmanB
the legendBpartyBzeroBonBlegacyB2002Bthe simsB3dB
revolutionBislandBleagueBtalesBstoryBghostBdanceB2004BprojectBlastBtigerBcryBassaultB2003BxBthe gameBmega manBfireBarcadeBin theBof warBfromBfootballBrallyBliveBkingdomB	episode 1BpgaBcombatBrockBpokemonBpga tourBncaaBmagicBwweB	spidermanBgolfBdragon ballBtonyBrisingBswordBsamuraiBpinballBkingsBgrandB2 theBwoodsBtennisBsoulsBsagaBdungeonBstormBpotterBmoonBharry potterBharryBearthBatBvBtelltaleBsportsBopsBnoBgearBfighterB
tony hawksBmarvelBhawksBf1BzB	woods pgaBtiger woodsB	simulatorBroadBoriginsBhunterBgamesB2005BtaleBrevengeBfifa soccerBfallBcrashBsecretBpiratesBmlbBlightBgame ofBeliteBdynastyBdawnBseasonBmaxB	episode 3BdeluxeBborderlandsBbeyondBstrangeBrogueBplanetBknightB	episode 2BcurseBclubBbattlefieldB07B	warhammerBpro evolutionBpowerBnarutoBevolution soccerB	challengeB	unleashedBprinceBfalloutBdynasty warriorsBcompleteBage ofB
video gameBvideoBtotalBsteelBsegaBpuzzleBnba liveBanniversaryBvrBto theBthe telltaleBthe lastBnascarBmutantBlife isBiBescapeBdisneyBdestinyBaceB2001B06BwarriorBtombB
is strangeB
definitiveBchaosBallianceBadvanceBworldsBusBtelltale seriesBstrikeBofficialBironBempireBeffectBwildBthe officialBoutBmyBmasterBlordBfarB08BwarfareBtacticsBparkBmanagerBheartsBdevilBcommandBallB6B	operationBmenBlittleBfightersBball zBbackBalienB10BteamBsilentBscrollsBsamB
potter andBmortalBlord ofBhonorBhitmanBgodB	episode 4Bdefinitive editionBdeathBcutBcapcomB
battle forBtrilogyB	the elderBstreet fighterBsniperBshadowsBpacmanBjourneyBelder scrollsBelderBdarknessBbigBtales ofBspecialBringsBorBjustBhotBdragonsBdoubleBbandBthe lostBthe darkBraiderBmonkeyBknightsBfutureBendBcarsBbladeBartBarmyBwormsBtransformersBtomb raiderB	the ringsBthe lordBracerBpackBorderBmortal kombatBkombatBfateBdisneysB
batman theB7B2008B	videogameBunderBitBguitarBfightBempiresBedgeB	the worldBraymanBrainbowBodysseyB
metal gearBjamBguitar heroBexBespnBbrokenBautoBzombieBshowBrideBrageBpartBof honorBmemoriesBmajorBmadnessBhellBfar cryBdsBdriveBdirtBcastlevaniaBarenaBreconBnBmovieBmajor leagueBlordsBii theBgodsBghost reconBdawn ofBcoldBclancys ghostBattackBaliveB8ByouBwrathBwithinBtycoonBthunderBmassBkongBkingdomsBextremeB
duty blackB	black opsBatelierB09BxmenBwrcBworld ofBupBtrekB
tournamentB	total warBthronesB	star trekBskyBskaterB	prince ofBplusB	of heroesBmotogpBmedal ofBmedalBhorizonBfuryBfullBfallenB	episode 5BdoomBdayBcurse ofBboyBwolfBwithBwars theBthe showBsuper marioBsplinter cellBsplinterBsixB
sid meiersBsidB	rock bandBrawBpersiaBparadiseB	of persiaBncaa footballBmodernB	minecraftBmeiersBmay cryBmayBmass effectBleague baseballB	heroes ofBharvestBgalaxyBfearB	devil mayB
dark soulsBcomplete editionBclancys splinterBcellB007BzeldaByakuzaB
world tourBwinterBwhiteBtokyoB	the videoBthe kingB	smackdownBshowdownB	shadow ofBroad toB	nightmareB
legends ofBfrom theBclassicsBblueBadventures ofB2012Bwarhammer 40000BuniverseBtheftB
the secretBseaBreturnBraceB
pro skaterBof zeldaBof timeBmarvelsB	hawks proBgreatBdogsBcontrolB	championsBbeatBatvBarmsB40000B3 theB11Bwwe smackdownBworld rallyBwitchBvirtuaBtopB
theft autoBstarsBsoulB	shippudenB
redemptionBpieceBphoenixB	one pieceBmysteryBmonstersBking ofBinfiniteBhillBgrand theftBdriverBdragon questBdead theBcupBcrystalBcoreBchapterBarkhamBanBa newB9B2009B20BwayBundergroundBtetrisBsunBspecial editionBsaintsBrunBrevelationsBrally championshipBrainbow sixB
pirates ofBoldBnightsBnaruto shippudenBgoBenemyBdirectors cutB	directorsBdieBdaysBclassicBclancys rainbowBblitzB
apocalypseB2010B2006Bworld championshipBvs rawBsuitBstoneBsolidBsmashBsmackdown vsBsiegeBshinBof fateB	lego starBjusticeBgundamB
gear solidBfrontierBeternalBenhancedBelevenBdungeonsBdefenseBcrimeBagesB2011B13ByourB	world cupBwestBwarlordsB	unlimitedBultraBultimate ninjaBtrialsBthirdBtenseiBtekkenBspyroB
saints rowBrushBrumbleBrowBremixBremasterBpersonaBoverBoriginalBolympicB
of thronesBmxBmissionBmeBmarchBlaBiceBheatBharvest moonB	guardiansBgenesisBcrownBconflictBcodeBattorneyBamongBaliensB	2 episodeBwolfensteinBtitanBthreeBsoldierBsniper eliteBsims 2Bsilent hillBshin megamiBsam maxBpursuitBpokerBout ofBmegami tenseiBmegamiBmachineB
just danceBjediBjamesBhardBgateBfriendsBfreedomBfirstBdreamsBdestructionBdead orBchronicles ofBburnoutBbatman arkhamBancientBamericanBalphaBaloneBairB2014B12BzoneBwantedBsquadB
skylandersBsamurai warriorsBprincessBphantomBmuseumBmaniaBlandBjoeBjackBhomeBhaloBevil revelationsB
dragon ageBdonkeyBdead risingBdeB	commanderBcivilizationBcastleB2013B14BzombiesBwrath ofBweBwatchBvolBvegasBturtlesBthe newBthe deadBsinB	shatteredBsevenB
season twoBromeBor aliveBof lightBmodeBmightBkartBiii theBheartB	freestyleBfiaBenhanced editionB
darksidersB	dangerousBconquestBbrothersBbrainBbmxBblastBagentB18ByugiohBysBxtremeBwwiiB	wrestlingBwiiBwarcraftB	vs capcomBvalhallaBturboB	the thirdB	the greatB	the finalBthe caribbeanBtestBteenageBshrekBsecondBroundB	rings theBresurrectionBpointBplayBpathBpacificBof twoBninja gaidenBmonster hunterBmodern warfareBmeiers civilizationBinfinityBimpactBhouseBgridBgoldenBgaidenBforzaB	fia worldB	featuringBfactionBdontBdonkey kongBdeepBdeadlyBcursedBcrisisB	caribbeanBbyBbrosB	bombermanBbioshockB	awakeningBarmoredB64BufcBtwo episodeBtowerBthisBteenage mutantBsuper monkeyBseriousBreturnsBreduxBred factionBrainBpoolBorochiBof fightersBninja turtlesBmutant ninjaBmonkey ballBmindBlords ofB	legendaryB	legacy ofBincBhumanBhulkBhd remasterBgearsB	forgottenBfall ofBdreamB	destiny 2BconquerBcollegeBbeforeBbadBamong usBactionBace attorneyB	wastelandBvampireBtrailsBthrones episodeBthe simpsonsBthe moonB
story modeBskiesBsims 3BsimpsonsBshippuden ultimateBshiftBserious samB
resistanceBof fireBof darknessBnamco museumBnamcoBmoreBmirrorBminecraft storyBmayhemBjonesBinternationalBhighBgunBguiltyBforeverBforcesBfootball managerB
fifa worldBfamilyBeaBdxBdisgaeaBdeusBdance revolutionBdance danceBcouncilB
conspiracyBcompanyBcommand conquerBcloneBcitiesB	anthologyBanniversary collectionBallstarsBallstarBadvancedBacademyB2000B19BwitcherBwingsBwars battlefrontBvs atvBvietnamB	vengeanceBtoyBtitansBthe witcherB	the forceBthe fallB
the dragonBthatB
test driveB
tales fromBstoriesBspyB	spongebobBspiritsBsoulcaliburBsnkBsingstarBrocketBracersBpuyoBprophecyBpath ofBof warcraftBnorthBninja stormBmx vsBmetroBlegacy collectionBkingdom heartsBinsideBindiana jonesBindianaBhuntBguilty gearBguardians ofBgod ofBfuBfrontBfourBfortuneBfatalBfableBevolvedBdeus exB	detectiveBdangerBcommandoBclankBcity ofBchildrenBcauseBbraveBbountyBbattlefrontB	battle ofBback toB
armageddonB
ace combatByearBxiiiBxcomBx2Bwarriors orochiBvolumeBvalleyBtropicoBthe wolfB
the cursedB
supercrossBssxBsims 4Bsherlock holmesBsherlockBshantaeBseasonsBsceneBsacredB
revenge ofBreloadedBratchetBrabbidsBpureB	pixeljunkB
persia theBon theBnukemB
nightmaresBnextBnetworkBneoB	motocrossBmonkey islandBmightyBmidnightBmastersB	marvel vsBmafiaBloveBlineBlairB
incredibleBimmortalBhoopsBholmesBheroes 2Bhd collectionBhandBgravityBfire emblemBfarmingB	everquestBemblemBduty modernB
duke nukemBdukeBdisneypixarB
dishonoredB
dimensionsBdevilsB
dead spaceBcrash bandicootBconsoleBchickenBcaptainB	bandicootBbaldurs gateBbaldursBat theBarmored coreBabyssB2k3B2016BzooB	wars jediB
vs zombiesBversionBultimate editionBtrueBthe oldB	the northBthe incredibleB
the galaxyBthe councilB	the cloneB
terminatorBtagBswordsBstreetsBstrange episodeB	strange 2BsquarepantsBspongebob squarepantsBspiritBsouthBsongBsoldiersBsniper ghostBsilverBseries episodeB	secret ofBrebornBratchet clankBorder ofBolympic gamesBofficial videoBoddworldBnotBnintendoBmario partyBmansBmanaBkirbyBkinectBjurassicBislandsBhedgehogBgloryBghostbustersBghost warriorBgalacticBflightBfilesB	ea sportsB
dreamworksBdivisionBdesertB	defendersBdead seasonBdead episodeBdcBcrysisBcrimsonBcrewB
clone warsBclashBboxingBbookBbondBblazblueBbattlegroundsBashesBart ofB2019B2018B2007B1 theByorkB
wolf amongBwhatBvirtua tennisBviiBviBurbanBuefaBtwistedBthe originalBthe hedgehogBthe endBthe chroniclesBthe borderlandsBsurvivorB
steamworldBstateBspinBsouls iiB
series theBscrolls onlineB	sacrificeBrpgBroomBrevelations 2BrealmsBpredatorBportalB	plants vsBplantsBoceanB	nfl blitzBnew yorkBnew frontierBneptuniaBmvpBmobileBmidwayBmedievalBmarvels guardiansBmadBlongBlegionBleftBlawBkung fuBkungB
just causeBjudgmentB
james bondBiron manBinvasionBinvadersBgunsBgoldBgirlsBghostsBfight nightB	fallout 3BfactoryBeyeBedenBduelsBdrBdivinityBdigimonBdiabloBdestroyBdefenderBconsole editionBchildren ofBchildBcatBcaseBbudokaiBbridgeBbrawlBbond 007BbittripB
beyond theB
before theBbannerB2020B2017B2015B15ByearsBxivBxiB
wright aceBwrightB	world warBworld seriesBwillBwar ofBvoidBvanB
us episodeBtwilightBturokBtriggerB	treasuresBtreasureBtrainBtouchBtomorrowBthree kingdomsBthe wildBthe olympicB	the movieB	the firstB
the fallenBtale ofBsurvivalBsuper heroesBstate ofBslugBskateBseries aBrugbyBrepublicB	reckoningBrealmBquantumBpuzzle questBprimeBpixelBphoenix wrightBpayneBouterBopenBofficial gameBof ironBof evilB
of empiresBof aB	new worldB	new vegasBnesB
ncaa marchBmrBmode episodeBminiBmiddleearthBmetroidB
metal slugBmercenariesBmax theB	max payneB	master ofBmarch madnessBmachinesBlikeB
lego harryBhiddenBheroes trailsBheavyBguardianBgoneB
generationB	gatheringBgalaxy episodeBfastBfallout newBextendedBeuroBeternityBenterBenemy withinBeffect 2Bearth defenseBduskBdownBdemonsBdemonBdeltaBdefense forceBdead islandBdashBcrossingBcouncil episodeBchampionship editionBbuzzBbrotherhoodBblazingBbattlesBarmy menB	alchemistB2k8B2k7B2k6B17BzenB	z budokaiBwinning elevenBwinningBwhoB
watch dogsB
warfighterBwar iiBwallaceBv skyrimBunrealB
undercoverB	unchartedBultimate allianceBuBtrineBtrackBtop spinBtinyBthiefBthe tombB	the stormB	the orderBthe nightmareBthe gatheringB
the futureBthe forgottenB	the enemyB
the devilsBthe darknessB
the bannerBtankBsystemBsword ofBswitchBstrongBstolenB	speed hotBspace invadersBsnowboardingBskyrimBshooterBshaunBsenran kaguraBsenranBsecretsB	scrolls vBsaveBsandsBroyaleBridge racerBridgeBrhythmB	return toB	return ofBretroBrescueBreignBquakeBpreyBpirateBphantasy starBphantasyBpeopleBpastBpaperBonesBoffroadB
of shadowsBof mightB	of juarezBnexusBneverwinterBneverBnemesisBmystery dungeonB
mysteriousBmysimsBmurderB	might andBmidway arcadeBmidnight clubBmemories ofBmarvel ultimateBlost planetBlost inBlinkBlego batmanBknockoutB
knights ofBkillerBkaguraBjuarezBinazuma elevenBinazumaBin timeBin armsB
impossibleBice ageBhot pursuitBgroundBgothamB	golf clubBglobalBgetBgenerationsBfrontier episodeBforza horizonBforestBfor theBflatoutBfiveBfistBfishingBfinal seasonB
fighter ivBfeverBfarming simulatorB	fantasticB	fallout 4BfairyB	enter theBdungeons dragonsBdividedBdestroy allBdeluxe editionB	deceptionBcry 4BcorpsBcoolBcontraBcolorsBcollege hoopsBburstBbrothers inBbroken swordBbreedBbreakBborderlands 2Bbanner sagaBavatarBat warB	ascensionBanimalB	and magicBadventures episodeB4 theB21B2 aB0Bzen pinballBwondersBwheelsBversusBvan helsingB	uefa euroBtunesB
trackmaniaBtower ofBtiesBthis isB	the threeB
the silverB	the sevenBthe planeswalkersBthe parkB	the nightBthe incrediblesBthe divisionBthe completeBthe adventuresBterrorBtaxiB	syndicateBsurferB
streets ofBstealthBsquadronBspiderman 2Bspeed undergroundB
speed mostB	sonic theBslugfestBscribblenautsBruneBrollercoaster tycoonBrollercoasterBrock ofBrivalsBringBred deadBrebirthBrealBrampageBpresentsBpower ofBpoliceBplaystationBplaneswalkersB
pillars ofBpillarsBpetsBpart 2BpanzerB
overcookedBoriginBonimushaBomegaBoffB
of seasonsB	of monkeyBof manaBnoireBniohBneonB
nba streetBnascar heatB
nancy drewBnancyBmvp baseballBmustBmummyBmost wantedBmostBmlb slugfestBmiamiBmen ofBmemoryBmax episodeBmarvel superBmarsBmarkB
mario kartBmansionB
man battleBmagicalB	magic theBlooney tunesBlooneyBlego indianaBlarryB	labyrinthBkaraoke revolutionBkaraokeBjungleBjetBisleBincrediblesB	immortalsBi amBhyperBhumansBhowBhourBhelsingBhawkBhappyB
guacameleeBgtBgears ofBfroggerBfoxBforsakenBforce unleashedBfitnessBfightingBeuropaBepisode oneB	episode iBepicB	encounterB	emergencyBedge ofBechoesBduels ofBdrewBdinoB	dimensionBday ofBdaveBcsiBcrown ofB	creed iiiBcrazyBcold warBcoastBclash ofBclancys theBchaseBcarnivalB	bustamoveBborderlands episodeBbook ofBbindB	bayonettaBbattlefield 3B
basketballBbaseball 2004B
avatar theB	attack onBatlantisBareBarcade treasuresBarcBanotherBanniversary editionBamazingBamBallstar baseballBafterBacesB4x4B16BxxBwolvesBwithoutBwithin episodeBwipeoutBway ofB	wariowareBvs theBvol 2BvalkyriaBus navyBuniversalisB
underworldBunderground 2BunboundBty theBtyBtvBtronB	trails ofBtour deB
torchlightBthroneBtheoryB	the witchBthe ultimateBthe tasmanianBthe sunBthe samuraiBthe golfBthe godfatherBthe evilB	the ashesBtenchuBtasmanian tigerB	tasmanianBtakBtailB	sword artB
suit larryBsuit gundamBstyleBstrong badsBstrikerBstrange beforeBstationB
south parkBsonsBsonic atBsong ofBsocomBslayerBsistersBsimcityBshovel knightBshovelB
shadows ofBsealsB	scoobydooBruinBromanceBrobotBriverBrenegadeBremainsBravingBrBquest chapterBprixBpotter yearsBpeople episodeBpark theBpacman worldBoverlordBoutlawBof steelBof spyroBof manBof eternityBof destructionBof coldBof agesBno moreBnineBnight roundBnierBnickelodeonB
nes seriesB
navy sealsBnavyBnationBnascar thunderBmobile suitBmirrorsBmirraBmickeyBmaskBmario sonicB
man legacyBmamaBloneBleisure suitBleisureBleBlaraBla noireBkings questBkillBixBi iiBhuntersBhow toBhitsBheistBhand ofBguildBgreenB
grand prixBgoodB	godfatherBghostbusters theBgauntletBgardenBgame forBfusionBfriendBfranceBfor attractiveBfifa streetBfantasy xivBfB	expansionBevil 5Bevil 4Beuropa universalisBendlessBencoreBeffect 3BduelBdigB	de franceB
dave mirraBcry 5Bcry 3Bcrime sceneBcreed valhallaBcountryB	cool gameBcookingBcolonyB
cold steelBclassic nesBchromeBcarnageBbrigadeB	breath ofBbreathBbowlBborderlands theBblockBblobBbirdsBbelowB	beginningBbattle networkB	bads coolBbadsBavengersBattractive peopleB
attractiveB
art onlineBarkBangelsBangelB
all humansBalien breedB
alchemistsBage originsBafricaBactB
a telltaleB5 theB4 deadB2k5B2k2Bzombie armyBxmen theB	world theBworld soccerB	witch andBwinter gamesBwars pinballB
warriors 5B
warriors 4B	warfare 2BviiiBvelocityBuprisingBunknownB	under theB
under fireBturismoBtruckBtownB	to rightsBto lifeBtmntB	ties thatBthereBthe videogameBthe skyBthe redB	the powerBthe lionBthe legoBthe escapistsB	the earthBthe ancientBthe amazingBthe adventureB	that bindBsyberiaBsurviveBsupremeB
superstarsBsupercross theBsuper streetBstory ofB
star oceanB
spy hunterBspectrumBsoccer winningBsilenceBshodownBshiningBshantae andB	sega agesBseason episodeBscene investigationBsave theBsanBsaga 2BrunnerBruinsBrouteBround 2BrootB
romance ofBrisenBrightsBridersBremnantBreign ofBraving rabbidsBrace driverB	quest forBpunchBproject carsB	professorBportableBpocketB	pinball 2BpicrossBperfectBpanicB
painkillerBpainBorcsBoperation flashpointBolympic winterBofficial videogameBof wwiiBof wrestlingB	of shadowB	of narniaBof magicB
of fortuneBof fearB
of destinyBobscureBninjasBnight ofBnhl hitzB
nfl streetBnfl 07Bneverwinter nightsBnarniaBmystBmovesB
motorsportBmonster energyBmoneyBmissingBmirra freestyleBmineBmiku projectBmikuBmight magicBmaximumBmatrixB
mario brosBmakerB
madagascarBluminesBluigiBlittle nightmaresBlionBletsB
lara croftBkings bountyBkillingBkatamariBitsBisle ofBinvestigationBinfernoBhockeyBhitzBhistoryBheavenB	hearts ofBhawks undergroundBhatsune mikuBhatsuneBguyBguardian ofBgrooveBgrimBgran turismoBgranB	goldeneyeBgodzillaBgeneralBgeminiBfridayBfreestyle bmxB
flashpointBflameB
fighter iiBfieldBfantasy crystalBexpressB
experienceBevil withinBeveBetrianB	escapistsBescape fromBenergy supercrossBenergyBemBechoBeaterBdustBdoorB	dodgeballBdelta forceBdefianceBdef jamBdefBdead toBdead redemptionB	cybertronBcthulhuB	csi crimeBcrystal chroniclesBcrusadeBcroftBcreed chroniclesBcreatureB	contractsB	commandosBcloseBcivilizationsBcircleBchannelBchampionship pokerBcastlevania lordsBburningBboxBboomB
bloodrayneBblindBbladesBbioshock infiniteBbikiniBbananaBback inBaxiomB	attack ofBarlandBarkham originsBapeBallstars racingBaboutB3 wildB2ndB
zoo tycoonByookalayleeBxxlBxvBxtreme legendsBxmen legendsBxlB	witcher 3Bwings ofB	wild huntBwerewolfBwaveBwatersBwars republicBwars episodeBwars 2Bwarriors ofBwarriors gundamB
warriors 2BwarlockBwakeBvol 1BvictoryBvalkyrieBvalkyria chroniclesBuntamedBunreal tournamentB
two worldsBtwo theBtwinBttB
true crimeBtrooperBtripleBtripBtrickyBtrialBtokiBtimesBthroughBthrillvilleBthievesBtheyBthe westBthe warriorsB
the titansB	the starsB
the shadowB	the outerB
the matrixBthe longB	the hedgeBthe edgeBthe directorsB	the curseB	the blindB	tenkaichiBtempleBteam racingBtableBsuperhotB
super megaBsummonerBsummerB
strongholdBstrikersBstreet racingBstopB
steinsgateBsteamworld digB	squadronsB
sports ufcBspeed carbonBsleepingBshotsBshinobiB	series ofBseasBsamurai shodownBrussiaBroyalBrow theBrow ivB
route zeroBrogue trooperBrockyBrising 2BriderB	red alertBrecordBrecon advancedBrebelBravenBraidersBradBquizBqubeBpsychonautsB	prototypeBproject divaBprisoner ofBprisonerBprimalBpopBplaceB
pinball fxB	persona 4BpaydayBpark baseballBpandaBpacman championshipBoverdoseBover theBoutlastBouter worldsBori andBoriBops iiBon titanBoff theB
of wondersBof vanBof usBof sinBof rockBof newBof memoriesBof chaosB	of arlandB	of amalurBoddyseeBnighBnfl 06B	new superBnba jamBmxgpBmust dieBmusicBmore heroesBminutesBmercuryBmechBmcrae rallyBmcraeB
masters ofBmark ofBman zeroBman xBlivesB	lightningBlibertyB
liberationBlego theBlego marvelB	layers ofBlayersBkingdoms ofBkillzoneBkentucky routeBkentuckyBjurassic worldB
journey toBjedi knightBjakBjaggedBisland chapterBis theBis nighBintoB	injusticeBincredible hulkBimmortals fenyxBii riseBii crownBhyperdimension neptuniaBhyperdimensionB
hot wheelsB	hot shotsBhopeBhoodBhollowBhitman episodeB	high heatBhelpBhedgeB
heat majorBhavocBhasbro familyBhasbroBhalflifeB	guerrillaBgrandiaBgothicBgold editionBgoingBgirlBgiantsBgermanyBgeometry warsBgeometryB
game nightBgalactic civilizationsBfxBfu pandaB	fracturedBforza motorsportBfortressBfordBflight simulatorBfist ofBfenyx risingBfenyxB
fantasy xvBfamily gameBfaction guerrillaB
extinctionB
expeditionB	evil deadBetrian odysseyBepisode twoBepisode iiiBend ofBend isBelB	echoes ofBdungeon siegeBdreddBdragons dogmaBdoom eternalBdoom 3BdogmaBdivaB
diablo iiiBdescentBdefender ofBde blobBdarkestBdark allianceBdanganronpaBdaB	cyberpunkBcyberBcrusaderBcrossBcriminalB
crazy taxiBcooking mamaBconquer redBconanB
company ofBcolin mcraeBcolinBcoachBcloudBclassics collectionBchristieBchild ofB	chapter 1Bchaos theoryBcaveBcaribbean theBcardBcarbonBbullyBbreakerBbloodstainedBblazing angelsBbionic commandoBbionicBbetweenB	bejeweledBbeeBbeachBbattlefield badBbaseball 2003Bball bananaBbad companyBatariBarthurB	art styleBarmy ofBangryBanarchyBamericaBamazing spidermanBamalurBalone inBalive 5BalertBalanBagatha christieBagathaB	aftermathBadvanced warfighterBa taleB51B
40000 dawnB2k21B2k10B25B1 2Bzero escapeByoshisByokai watchByokaiBxiiBwrestlemaniaB
worlds endBworld isBworld inBwonderB	wild westBwhereBweaponsBwars iiB
warriors 8BwardrobeBwandererBwallace gromitsBvrallyB
volleyballBvirtua fighterBvillageBvikingsBviewtiful joeB	viewtifulBv2BuntoldBunravelBunityB
undisputedBufc undisputedBtt isleBtribesB	trials ofB	toy storyBtowersBtour 07BtouhouBtormentBtoriBtooB	toki toriBto gloryB	titanfallBtimesplittersBtheaterB	the whiteBthe wardrobeBthe vampireBthe twoB
the returnBthe phoenixBthe mysteriousB	the mummyBthe missingB	the magicB	the houseB
the goldenB
the gobletBthe crewB
the battleBthe awakeningBthe apocalypseBthe alchemistBthanBtensei devilBtennis worldBtanksBtakeBtag teamBtacticalBsymphonyBsurgeB	sufferingBsuddenB
subnauticaB	sturmovikBstreet 2BstrangerBstorm episodeBstickBstarringB	starcraftBstar foxB
star forceBsquarepants battleBspyro aBsporeBspiderman 3B
spellforceBspeed prostreetBspecial forcesB
space hulkB	souls iiiBsolid vB
soldier ofBsocom usBslyBsleepBslamBskylinesBskateboardingB	six vegasBsightBshopBshogunBshieldBshenmueBshellBshaun whiteBsetBsea ofB
scrolls ivBsavageBrune factoryBrtypeBroadsBritualBrising 3BriftBreportBremakeBrayman ravingBrayman 3BrailsBraider underworldBragingBracing 2BrabbitB	puyo puyoBpulseB	prostreetBprofessor laytonBpro bmxBprisonBpressureBpokemon mysteryBplayhouse episodeB	playhouseBpirate warriorsBpilgrimBpiece unlimitedBpiece pirateB	paintballBoutcastBoutbreakBosirisBoriginal trilogyBoriginal sinBoriginal adventuresBops iiiB
operationsBon tourBokamiBofficial motocrossBof rageB	of osirisBof kingsBof bloodBobserverBnothingB
nintendogsBncaa collegeBnba ballersBmythBmotocross videogameBmonster jamBmonopolyBmlb theBmissionsBmiddleearth shadowB	microsoftBmicro machinesBmicroBmichonneBmetroid primeBmeltdownBmega baseballBmat hoffmansBmatBmarvels avengersBmario luigiBmarineBman starBmafia iiBlongestBlondonBlightsBlethalB
lego movieBleft 4Ble tourBlaytonBlast airbenderBkong theBkong countryB	kombat 11BknowBkidBjustice leagueBjurassic parkBjujuBjuicedB	jones theB	jones andBjointBjack theB	its aboutB	invisibleBinnerBincredible adventuresBin blueBil2 sturmovikBil2Bii definitiveBhuman revolutionBhouse ofBhospitalBhoffmans proBhoffmansBhitman 2Bhearthstone heroesBhearthstoneBheart ofBheadBhawxBhawks americanBharveyBharmonyB
guild warsBgromits grandBgromitsBgrand adventuresB	goblet ofBgobletBgiantBgiBgermany 2006Bgathering duelsBgatesBfx 2B
future theBfuriousBframeB
for bikiniBfloorBflamesB	final cutBfifa 07Bfear 2B	fantasy xBfantasy viiBfameBeyesBexodusBexitBex humanBevoBevil 7Bevil 3Bevil 2BeveryBeuropeanBeuropeBeragonB	empire ofBelite v2BdyingBduty ghostsBduty advancedBduckBdubBdualBdrive unlimitedBdrawn toBdrawnBdraculaBdouble dragonBdouble agentB	dmc devilBdmcBdjB
division 2Bdivinity originalBdisney infinityB	discoveryBdiscoBdiceBdevils playhouseBdesert stormBdead michonneBdarksiders iiBcup germanyBcreed iiBcounterstrikeB	continuesBcontactBconflict desertBcomplexBcolossusBcobraBcliveBclancys hawxBcities skylinesBchronoBchaptersBchamberBchainsBcell doubleBcatchBcastlestormBcars 2Bcapcom classicsBburnBbulletBbuildersBbubbleBbrightBbravoBbottomBborderlands 3Bbmx 2BbluesB
blood bowlB
blitzkriegBblair witchBblairBbirdBbionicleB	biohazardBbikini bottomBbeBbattlefront iiBbassBbangBballersBatlasBastroBassassinBaroundBariseBarchivesBarB
ape escapeBanimal crossingBamnesiaBaliceBairborneB	airbenderBage iiBadventure timeBadvanced warfareBactiveB
about timeB8bitB7 biohazardB2k9B2021B1942B	07 soccerBzone ofByoursBxenoblade chroniclesB	xenobladeB
xcom enemyBxboxBx2 hdBx x2BwwfBworld atBworld 2B	wonderfulBwolfenstein theBwizardsB	without aBwith youB	with loveBwindB	where theBwetBwerewolf theBwatchmen theBwatchmenBwarzoneBwars iiiB
warriors 3Bwarrior contractsB	warrior 2Bwar warhammerBwar forBwalleB	vs donkeyBvinciBvikingBvergeBvaultB	vanishingBvalorBurban trialBunwritten talesB	unwrittenBunmaskedBunder pressureBundeadBtycoon 2Btwisted metalBtruthBtransformers theBtransformers revengeBtrainingBtour 08Btour 06B	tormentedBtogetherB	to borutoBtitan questBtime crisisBthorBthingsB	the wrathBthe warlordsB
the templeB	the swordB	the surgeBthe sufferingB
the stolenBthe sorcerersBthe roomBthe riseBthe revolutionB
the policeBthe pathBthe pacificB	the ninjaB
the moviesBthe longestBthe legendaryBthe gungeonBthe fracturedB
the endersBthe dungeonBthe deathlyB	the crownBthe cityBthe bookB	the blackBthe bigsBterminator 3Btensei personaBtennis 2B	temple ofBtelltale gamesBsyphon filterBsyphonBswatBsupermanB	superbikeBsuper smashB
super meatBsumBsuikodenBstyxBstriderBstealth incBstardustBstaffBspiderman friendBsphinxBspelunkyBspeed undercoverBspectrum warriorBspartanBspace programBspace 2BsoundBsorcerers stoneB	sorcerersBsolarBsoccer 2009Bsoccer 2008Bsoccer 2005B	soccer 09B	soccer 08B	soccer 06BsnakeB	smoke andBsmokeB
smash brosBslimeBskater 3Bskater 1Bsix 3BsithBsinsBsinkingB	sine moraBsineBsimpsons gameBsiege ofB
shots golfB
shellshockBsharkBshaolinBshankB	shadowrunBshadow warriorBsega superstarsB
sega rallyB	section 8BsectionB
secrets ofBscott pilgrimBscottBschoolBscene itBscarletBsargesBsands ofBsanctumB	sanctuaryBsan franciscoBsamurai jackBsaltBsaga 3BsBryzaBruinerBromancing sagaB	romancingBrobotechBriotBride onBretributionB
remotheredBrememberBredoutBreachBrayman originsBrayman legendsBraw 2009Braw 2008BratatouilleBrally 20Braider legendBrage 2BradioBquest challengeBqueenBprogramBpro cyclingBport royaleBportBplaygroundsBpixeljunk monstersBpitfall theBpitfallBpinball hallB
pilgrim vsBpikminBpictures anthologyBpicturesBpeter jacksonsBpeterB	persona 5BpeggleBpeaceBpayday 2BpaybackBpaper marioBpandoraBoverrideBoverkillBoutrunBoutlawsBor foeBoffroad furyBof unwrittenB
of thunderB
of secretsBof ruinBof romeB	of rhythmBof lifeBof hellBof fameB	of dreamsBof discoveryB
of cthulhuBof apocalypseBof anBobelixBnukem 3dBnowhereBno kuniBnintendo switchBnickelodeon kartBni noBniBnhl 06Bnfl gamedayBnfl 2005Bnfl 08Bnew beginningBnationsBnaruto ultimateB
narnia theBmtvBmoviesBmoveBmountB
motorstormB	morrowindBmoraBmirrors edgeBminigolfBmini ninjasBmichonne episodeB
men sargesBmelodyBmeat boyBmeatBmatterBmarvels spidermanBmario vsBmanhuntBman rideBmagnaBmagesBlynchBlunarBlocorocoBlive 06BlittlebigplanetBlion theBlegends theBledererB
layton andBlabBkuniB	king kongBking arthurBkart racersB
kane lynchBkaneBjobBjeremyBjames cameronsBjagged allianceBjacksons kingBjacksonsBiv theBiv blackB	island ofBinto theBinquisitionBinfamousBin bloodB	in actionBii turboBhunter worldBhoward ledererBhowardBhotline miamiBhotlineBhordeB	homefrontB
holmes theB
heavy rainBhazardBharborB	happinessBhallows partBhallowsBhall ofBhallBgunvoltBgungeonBgrowBgrid 2B	good evilB
golf worldBgoesB	god eaterBgi joeB	gate darkBgames seriesBgamedayBgame episodeBgalaxiesB	galacticaBfull spectrumBfuelBfrozenB	frontlineB	friend orBfrenzyB	franciscoBforgotten sandsBforce 2BfoeB	flashbackBflagBfireteamBfilterBferrariBfeaturing howardBfeBfatal frameB
fantasy xiBfame theBfaithBeyetoyBeye ofB	explorersBevil 6B
everythingBeverBespn nflBespn nbaBendersBemperorsBelementsBdying lightBdrivenBdrew theBdreamworks shrekB	dreamfallBdr marioB	dont knowB
dominationBdokiB
dogma darkBdogBdj heroBdisneys chickenB
dirt rallyBdirt 2Bdigimon worldBdevastationBdespairBdecayB
deathspankBdeathly hallowsBdeathlyBdaylightBdaxterBdartsBdark picturesBdark arisenBdamnedB	damnationBdamageBda vinciBcycling managerBcyclingB	croft andBcreed ivBcortexBcorsaBcorpseB	continuumBconstructorB	conflictsB	conditionB	company 2BcomeBcodenameBcoasterBcivilization viBcitadelBchronosB	chronicleBchosenBchicken littleBchessmasterBchessBchariotB
chamber ofBchallenge ofBcentralBcasinoBcars 3Bcaribbean atBcapcom 3BcameronsBbutcherBbushBburnout paradiseBbulletstormBbridge constructorBboundBborutoBboldBbloodyBblob 2BblackoutBblack mirrorB
black flagB	bind partBbigsBbeyond goodBbeastBbearBbattlestar galacticaB
battlestarBbattlefield 2B	battlecryBbakuganBazureBaxiom vergeBatv untamedBatv offroadBatelier ryzaB	at worldsBat seaBasterix obelixBasterixBassetto corsaBassettoBassemblyBasBarisenBareaBaragamiBaquaBapexBapartBaoBanomalyBanimaBangry birdsBalliedBalchemists ofBalchemist ofB	alan wakeBagainBa heroBa gameB76B57B50B5 noB40thB4 aB3rdB3d classicsB30B
3 ultimateB2k14B22B	2 unboundB2 battleB1 aBzxBzombies gardenB	zero roadByears 57Byears 14B	year zeroByear editionByakuza likeByaibaBxtreme racerBxmen originsB	xenoverseBxcom 2Bx2 wolverinesBx legacyBwrongBwrc 9B	worldwideBworld racingBworld 3BwonderworldB
wonder boyBwolverines revengeB
wolverinesB	wolverineBwizardB
within theBwingB	wild armsBwhite snowboardingBwesternBwellBweb ofBwebBwaterBwasBwars galaxiesB
warriors 6Bwarlords ofBwarioBwargameB	warfare 3Bwanted aBwallace gromitBwackyBvs snkB	vs aliensBviva pinataBvivaBvirtualBviceBveronicaB
vermintideBvendettaBvectorBvanquishBvampire slayerBv3BuponBunoBunitedBuniteBunitBunionB	unchainedBultimate spidermanBufoBtycoon 3Btwo thronesBtwo sonsBtwiceBtruckersBtroopersBtrivial pursuitBtrivialBtriple playBtrine 2BtravelerBtraumaBtrapBtransformers warBtransformedB
train yourBtour soccerBtop gunBtokyo xtremeBtoejam earlBtoejamB
to surviveBto edenBto dieBtitan 2Btides ofBtidesBthrillville offBthriller episodeBthrillerBthor godB
the zombieBthe yearB
the yakuzaB	the woodsBthe universeB	the towerB	the staffBthe spiderwickBthe sithBthe sinkingBthe serpentsB	the sandsB
the sacredB	the ruinsBthe ringB	the ravenB	the railsBthe prisonerBthe piratesBthe phantomBthe nextBthe manB
the littleB
the leagueBthe kangarooBthe impossibleBthe hundredB
the hobbitBthe halfbloodB
the grooveBthe forsakenB
the forestB	the flameBthe eyeBthe emperorsBthe deepB
the damnedBthe daBthe crystalBthe chamberBthe catB
the brokenB	the bladeBthe bindingBthe artBthe aftermathBthe 3rdBthe 13thBtetris worldsB	testamentBteslaBterrariaBtennis 3BteenBtearsBtangoBtakedownBtaito legendsBtaitoBswords soldiersBsword 5Bswitch editionBswingB
swap forceBswapBsurgeon simulatorBsurgeonBsurfBsupreme commanderB	supremacyBsuperstars tennisB	superstarB	superslamBsuperbike worldBsuper stardustBsummon nightBsummonBsudokuBsudden strikeBstuntmanBstuntBstrike forceBstreet 3B	strangersBstorm 4BstillB	stellarisBsteamworld heistBstarring mickeyBstarfighterBstar onlineBstaff ofBsquadrons ofBsprintBsports activeBspiderwick chroniclesB
spiderwickBspiderman webBspiderman theBspiderman shatteredBsparkB	spacetimeBsons ofBsonic maniaBsonic colorsBsonic allstarsBsonic adventureBsonBsolaceBsoccer 2012Bsoccer 2011Bsoccer 2010Bsoccer 2003Bsoccer 2002B	soccer 13B	soccer 12B	soccer 10Bsleeping dogsBskylanders swapBskylanders giantsB	sky forceB
skullgirlsBskinBskater 4Bskater 2BsirensBsins ofBsinking cityBsingBsimulator 2Bsilver surferBsilent scopeBsigmaBsideB	shrek theBshrek superslamBshrek 2BshootoutBshockBshattered dimensionsB
shark taleBshardsBshapeBseveredBserpents curseBserpentsBseries baseballBsega genesisBseekBsectorBsea episodeBscrolls iiiBscopeBscarfaceBsbkBsawBsabreBruins ofBround 3BrollB	rocksmithB	road rageB
river cityBriskBriddickBrhythm alienBrevenantB
retro cityBresortB	resonanceBremnant fromBreflexBredneckBrearmedBratsBrangersBrangerBrandomBrancherBrage 4BradiantBracing transformedBquest buildersBquest 2B
quantum ofB
quad powerBquadBpuzzle adventureBpuyo tetrisBprostroke golfB	prostrokeBprosB	project 8Bpro wrestlingBpresentsrunner2 futureBpresentsrunner2BpremonitionBpower racingB
power prosBpostalBplanet extremeBplanet coasterBplanBplagueBpirates curseBpinataBpiece grandBpenny arcadeBpennyBpenalBpearlBpataponBpassBpart twoBpart oneBpart 1BpantsBpandora tomorrowBovercooked 2BotherBorochi 3BorionBorigins wolverineBorigins blackgateB	orcs mustB
or nothingB	onslaughtBonimusha warlordsBones justiceBon fireBomenBokami hdBofficeB	of winterBof vengeanceB
of spiritsB	of solaceB
of riddickBof ninjaB
of nationsBof mindB
of midgardB
of libertyBof kayBof kainBof isaacBof finalBof decayB
of azkabanB
odyssey toBoddworld abesB	oceanhornBoblivionB
obelix xxlBnyBnutsBnot aB
north starBnormandyBnoirBnobunagas ambitionB	nobunagasBno timeBno mansBnitroBnioh 2Bnier automataBnhl 2005Bnhl 07Bnfl headBnfl 2004Bnfl 2002B
necromundaBnba shootoutBmy heroBmutha truckersBmuthaBmutant yearBmutant muddsBmuddsBmountainBmount bladeBmonsters vsB
monster ofBmonkeysBmondayBmojoBmodern combatBmode seasonBmmaB	mlb powerBmixBmission impossibleBmiracleB
millenniumBmidgardBmicrosoft flightBmichaelBmetro exodusB	mercenaryBmeleeBmegamixBmechwarriorBmcgrathBmarvel pinballBmario tennisB
mario golfBmankind dividedBmankindBmaneaterBmanager seasonBman 2BmakeBmaidBmagickaBmagic duelsBlost expeditionBlos angelesBlosBlockdownBlive 08Blive 07BlinksBlinesBlike aB	leviathanBlego piratesB	lego cityBlegionsB
legends iiBlegendary editionBleague heroesB	law orderBlakeBknockout kingsBkiwamiBkirbysBkingdom underBkilling floorBkickBkerbal spaceBkerbalBkeepBkayBkatanaBkao theBkaoBkangarooBkainBjumpBjrB
journey ofBjotunBjohnB
joe dangerBjak andBisaacBionBinvaders extremeBinstinctBinner worldBinfinite burialBindiaBin aBiii legendsBignitionBicewind daleBicewindBhyruleBhuntingBhundredBhoursB	homeworldBhobbitBhitBhide andBhideB	hero onesBhero iiiBherBhellsB
hedgehog 4B
head coachBhawks projectB	hawk downBhavenBhauntedBhatBhalfblood princeB	halfbloodBhail toBhailBhackguBgxBgroundsBgromitBgripB	great warBgradiusBgooseBgoldeneye 007Bgolden compassB	gods willBgoblinsB	gladiatorBghostlyBghost ofBgeniusBgate iiBgarden warfareBgalagaBgaiden 3BgBfuture legendBfunBfront missionB	from hellB
friday theB	four riseBforce blackBfor nintendoBfor cybertronBflyingBflowerBfloodBflatB
fistful ofBfistfulBfire proBfinal fightB	fighter xB	fighter 5B	fight forBfifa managerBfestivalBfencerBfatesBfate ofBfantasy xiiiBfantasy xiiB
fantasy ivBfantastic fourBfantastic 4BfantasiaB
family guyB
fallout 76BfalconBf1 2011Bextreme conditionBextraBextended editionBextendBexpansion packBexileB
ex mankindBevil hdB	evil codeBevil 0BevidenceB	everybodyBeverquest iiBericaB
episode iiBepic mickeyBenslaved odysseyBenslavedBengineBendwarBempire earthBemperorBelysiumB	elite iiiB	elementalBeclipseBeatBearlBeagleBdwarvesB	duty wwiiB
duty worldBduty 3Bduty 2Bdub editionBdryBdriftBdreamworks kungBdreams dontBdragons lairBdownhillBdoctrineBdoctorBdiverBdisneypixar walleBdisneypixar ratatouilleBdisney sportsBdisney epicBdisco elysiumBdirt 5B	diablo iiB	desperateB
desperadosBdeponiaBdefense gridBdeca sportsBdecaBdeathsBdeadly premonitionB	dead mansBdead aBdead 2Bdc superBdaytonaBdanger zoneBdaleBcyber shadowBcuriousBcrysis remasteredBcry 2BcrushBcrusader kingsBcrimesBcreed unityBcreed rogueBcreed originsBcreed brotherhoodBcreateBcovertBcourtBcossacksBcorpse partyBcornerBcontinuum shiftB
connectionB	condemnedBcompassBcommandos 2Bcommando rearmedBcombat racingBcombat assaultBcollection 2BcoffeeBcode veronicaBclub 3BclonesBclimaxBclancys endwarBcity undercoverBcity rampageBcitizensBchivalryBchimeBchestBcharlieB	chapter 2BcenterBcentBcell pandoraB
cell chaosBcatwomanBcatsB	catherineB	cat questBcastawayBcartelBcarnival gamesB	capcom vsBcapcom 2BcampaignB	burial atBburialBbundleBbulletsB	buffy theBbuffyBbudokai tenkaichiBbubble bobbleB
brothers aBbreachBbrassB	boyfriendBboy theBbowlingBbootyBboltBbodyBbobbleBblood dragonB	blitz theBblind forestBbleedBbleachBblazblue continuumBblasterB	blades ofB
blacklightB	blackgateB
black hawkBbittrip presentsrunner2Bbionicle heroesB
binding ofBbindingB	big muthaBbehindBbeginsB
battlezoneB
battleshipBbattle princessB	battalionBbashBbaseball 2005Bbaseball 2002BbanjokazooieBbandicoot 4Bball xenoverseBbalan wonderworldBbalanBbadassBazurBazkabanBawakensBawakenedBautomataBauto vBauto ivBatv quadBattorney trilogyBatomicB
assault onBassault horizonBasphaltBart academyBarrivalBarmiesBarkham knightBarkham cityBarea 51BarcticBarcanaBarcade classicsBarcade adventuresBantBangels squadronsBangelesB
and daxterB	ancestorsBamerican wastelandBambitionB
alliance 2B	aliens vsBalert 3BaladdinBakibasBair conflictsBagentsBage inquisitionBage 2BaegisBadventBabesBa dragonB9 fiaB50 centB4 modernB4 itsB	4 episodeB3 dubB3 chaosB3 brokenB2k20B2k19B2k18B2k11B2kB2 worldB2 videogameB	2 projectB2 petsB2 darkB13thB12 theB100B09 theB007 quantumBzombies battleBzombie apocalypseB	zero timeBzero actBzelda breathBzapperB	z kakarotB
yugioh theB	yugioh gxByuBys theB	ys originByour dragonB
youngbloodB	york cityByookalaylee andB
yonder theByonderByokus islandByokusByakuza kiwamiBxxxB	xx accentB
xiv onlineBxdBx tekkenBwwe allBwrestling iiB	wreckfestBwrc 8B	worms wmdB
worms openBworms 3dB	worlds iiB	world redBworld orderBworld evolutionB
world endsB	wolves ofBwolfenstein youngbloodBwolfenstein iiBwolf ofBwmdBwizardryBwith danielBwitcher talesBwinter sportsBwineBwindsB	windboundBwill ofB	will fallBwildsBwhosBwholeBwhenBwhatsBwhat remainsB
wet dreamsBwest ofB
welcome toBwelcomeB
weapons ofBwasteland 3Bwasteland 2B
warriors 7Bwarrior withinBwarhammer chaosbaneB	wargrooveBwarframeBwarfighter 2BwardBwar zB	war threeBwar romeBwar iiiBwar 2BwakeboardingBvs predatorBvs lovecraftBvs deathBvranBvolume 2BvirtueBvietcongBvictor vranBvictorB	vice cityBvesperiaBversus predatorB
veronica xBvanguardBvampyrBvampire theBvalkyrie profileBvaliant heartsBvaliantBvalhalla knightsBvalhalla editionBvalfarisBvacationBv4Bv2 remasteredButawarerumonoBusaBus theB	urbz simsBurbzBupon aBunruly heroesBunrulyBunravel twoBunlimited worldBunlimited 2Bunleashed iiB	universalBunfortunate eventsBunfortunateBunder siegeBunder nightBuncharted seasBultra streetBultimate marvelBultimate fightingBtzuBtysonBtypingB	two soulsB	two pointBtwistBtwilight ghostBtwelveB	turtles 3B	turtles 2Bturok evolutionBturning pointBturningBturbo hdBturbo championshipBtrue colorsBtroubleB	tropico 6Btrooper reduxBtron evolutionBtrine 4BtrickBtrials risingBtrial freestyleBtreasure trackerBtrauma centerBtransformers riseBtransformers darkBtrainerBtradingBtracksBtrackerBtrack fieldBtoyconBtoy soldiersBtournament 2B	tour 2k21B	tour 2007B	tour 2005B	tour 2004B	tour 2003Btour 2Btour 10BtornadoBtori 2Btorchlight iiiBtorchlight iiB	tony hawkBtomb ofBtokyo twilightB
tokyo 2020B	toca raceBtocaBtoad treasureBtoadBto trainB
to baghdadBtitan lordsBtime dilemmaBtime atBtiger 2BtideBthunder 2003BthumperBthrough theBthronebreaker theBthronebreakerBthis warB	third ageBthingBthimbleweed parkBthimbleweedBthe wayBthe warBthe wandererBthe vanishingBthe urbzBthe songB	the sleepBthe skinB	the siegeBthe shadowsBthe sexyBthe settlersB
the secondBthe seaBthe scienceBthe runB
the ripperBthe redemptionBthe reckoningB
the rapperBthe punisherBthe pastBthe necrodancerBthe mysteryBthe messengerBthe meltdownBthe masqueradeB	the knifeBthe journeyBthe jackboxBthe italianB	the innerBthe imperfectsBthe historyBthe hauntedBthe guardiansBthe ghostlyBthe gardensB	the floodBthe fellowshipBthe falconeerBthe experienceBthe eternalBthe dwarvesBthe duskB
the demonsBthe darksideBthe conspiracyBthe collectionB	the cloudB
the chosenBthe chocolateBthe caveB
the bunkerBthe bigB	the bardsB
the badassBthe arkBthe antBthe allianceBthe ageB	the abyssBthe abcBtexasBtetris effectBtestament ofB	teslagradBtesla vsB	territoryBterminator resistanceBtennis 4Btennis 2009BtenBtempestB
tekken tagBtalkBtales 2B	takes twoBtakesBtakenBtak theBtaikoBtag tournamentBtactics theB	table topBsyndromeBsymphony ofB	sword theBswarmBsurmaBsuperliminalBsuper turboBsuper soldierBsuper puzzleB
super heroBsuper bombermanBsummitBsubnautica belowBstrike suitBstrangers wrathBstrange trueBstory 3Bstorm iiBstorm 3BstitchBstick itBsteepBsteel titansBstarshipBstarpoint geminiB	starpointBstardew valleyBstardewBstarcraft iiB	star kensBstalkerBstackingBstacked withBstackedB
ssx trickyBssx onBssx 3Bspyros adventureBspyrosB
spyro dawnBsplitsecondB	spirit ofBspireB	spintiresBspin 3B
sphinx andBsphereB
spellbreakB	speed theBspeed shiftBspeed rivalsBspawnB	sparkliteBsparkleBspace 3Bsouth africaBsouls remasteredBsoulcalibur iiBsorrowBsorceryB
sonic segaBsonic ridersBsonic heroesBsonic generationsBsonic forcesBsolitudeBsolid 2Bsoccer 2004B	soccer 11B
snowrunnerB
snickets aBsnicketsB
snake passBsmash courtBsmallB
sly cooperBslugfest 2004Bslay theBslayBslainBskylanders spyrosBskullyBskies ofBsix lockdownBsirBsinner sacrificeBsinnerBsingularityBsin tzuBsin iiB	simulacraBsims inBsimpsons roadBsimpsons hitBsilver caseBsilent hunterBsilent assassinB
shiren theBshirenB
shield theB	shelteredBshelterB	shards ofBshantae halfgenieBshaman kingBshamanBsexy brutaleBsexyBseven sirensBsettlersBsessionsBsenuas sacrificeBsenuasBsengokuBsega allstarsBseedBsecret weaponsBsecret filesBsecret fairyBsecond sightBsealBscienceBscarlet nexusBscarface theBsarges heroesBsane trilogyBsaneBsam hdB	salvationBsakuraBsafariBsacrifice forBryza 2Brussia withBrunsBrunningBrumble boxingB	rpg makerBroyal editionBrow 2BrosesBroogooBrogue legacyBrogue agentBrodgersBrocket leagueBroboB
robin hoodBrobinBroarBrivalB	ritual ofBrisen 3Brise toBriptideBripperBrioB	rings warBrings conquestBrimeBride 4BriddleBrezBrevoB
revelationBretreatBresurrection ofBresistance tacticsBreservoir dogsB	reservoirBrereckoningBrequiemBrepublic heroesBrepublic commandoBreport everybodyBremastered collectionBremarsteredB
remains ofBreignsB
rehydratedB
regalia ofBregaliaBrefrainB	reelectedBreed thrillerBreedBredeemerB	record ofBrecon 2BrebootBreaperBrealm ofBrealityBrcBraw 2010BrapperB
rally revoBrainbow islandsBrailroadBraider anniversaryBraidenBragnarokBraging blastBrad rodgersB
race starsBquidditch worldB	quidditchBquest xiBquest heroesBquest galactrixBqube 2Bpuzzle fighterBpursuit remasteredB	pursuit 2BpunisherBpumpkin jackBpumpkinBpuddleBpsychonauts 2BpsychoBprotocolBprophetBproject warlockBproject gothamBprofileBprinnyBprincess madelynBpower rangersBpotter quidditchBportiaBpolarBpoker 2Bpokemon swordBpokemon rumbleBpoint hospitalBplus theB
planetsideB	planet ofBplanet alphaBpixeljunk shooterBpixel remasterBphogsBpersonal trainerBpersonalB	persona 3Bpersia warriorB	pdc worldBpdcB
party packBparappa theBparappaBpalsBpainkiller hellB
pacman andBoxenfreeBowlboyB	overwatchBover mutantBover europeBoutrun 2006B	outridersBoutlaw golfB	outlast 2Borochi 4Border upBorangeBops coldB	operativeBoperation surmaBopen warfareBonlyB	once uponBonceBomen 2B	olliolli2BolijaB	old worldBold republicBogreB	offensiveBof vesperiaBof valorBof unfortunateBof unchartedBof truthB
of thievesBof tanksBof soulsBof solitudeB
of shaolinBof resistanceBof powerBof pokerBof oldB
of mystaraBof mortaB	of mordorBof mineBof menB	of mayhemB
of madnessB
of legendsBof laBof judgmentBof illusionBof edithBof earthBof duskBof cobraB	of clonesBof championsB
of captainB
of camelotBof brassBof blackBof azurBof ariseBof animaB
of ancientBoddworld strangersBoddworld munchsBoceanhorn monsterBobscure theBnukem foreverBnomBnocturneBnobodyBno 9Bnintendo laboBnightwarB	nights atBnights 2Bnightmares iiBnightmare princeB	nightfireBnight inbirthBnidhoggBnhl 2k7Bnhl 2004Bnhl 2003Bnfl 25Bnfl 2003Bnfl 09Bnew dawnBnew colossusBneopetsBneon chromeB
neon abyssBnemesis riseBneighborvilleBnegreanuBnecrodancerB
ncaa finalBncaa basketballBnba playgroundsBnba 2k7Bnba 2k21Bnba 2k2Bnba 2k14BnavalBnaughty bearBnaughtyBnaturalBnaruto clashB
narita boyBnaritaBn saneB	n goblinsB	n forestsB	mythologyB
mystery ofBmystaraBmy timeB	my memoryBmy loveBmutant nightmareBmushroomBmuseum 50thBmuscleBmurdersBmunchs oddyseeBmunchsBmtx mototraxBmtxBmsB
mr drillerB
moving outBmovingBmovie videogameB
movie gameBmovie 2Bmoves streetBmototraxBmotorB	motogp 08BmotoBmotionBmortal shellBmortaBmordorBmoonlighterBmoon dsBmonster truckBmonster rancherBmonster houseBmonster 4x4B
monkeys ofBmonarchsBmogulBmirageBminority reportBminorityBminitB	mind overB
mike tysonBmikeB	mighty noBmichael jacksonBmetro reduxB
metro lastB	messengerBmercenaries 2Bmen andB	memory ofB	melody ofBmeiers piratesBmeetBmcgrath supercrossBmatt hazardBmattBmatchBmaster collectionBmassiveB
masqueradeBmask ofBmarvel nemesisBmario advanceBmario 3dBmarbleBmans skyB	manhattanBman ofBman 11B
make breakBmajestyBmageBmadelynBmachines v4BmachinaBluigis mansionBluigisBluckys taleBluckysB	lovers inBloversB	lovecraftB
lost soulsBlost legendsB	long darkBlone survivorBlondon 2012B	live 2005B	live 2004B	live 2003BlimboB	light theBlight drifterBliesBlemony snicketsBlemonyBlego worldsB	lego rockBlego jurassicB	league ofBleadBlateB	last stopBlast remnantBlast ofB
last lightB	larry wetBlandsBlamulanaBlair ofBlabyrinth ofBlabo toyconBlaboBkombat deadlyBknockout cityB	knight iiBknifeBkittyBkitBkingdom comeBkidsB	kens rageBkensBkeeperBkay anniversaryBkatamari damacyBkart 8BkakarotBkaiBjuju challengeBjudge dreddBjudgeBjuarez gunslingerBjotun valhallaBjones 2Bjojos bizarreBjojosBjoe theBjimBjewelBjettBjeremy mcgrathBjedi outcastBjawsB	jam steelBjadeBjackson theBjacksonBjackbox partyBjackboxBjackassBiv reelectedBitalian jobBitalianBit upBit toBit takesB	isolationBisland expressBis yoursBion furyB
invincibleBinteractiveB	innocenceB
innerspaceBinlineBinjustice 2Binfinite warfareBinfamyBindycar seriesBindycarBindivisibleBindigo prophecyBindigoBinc 2BinbirthB	in randomBin harmsB	in flamesBimpossible operationBimpossible lairBimportB
imperfectsBimmortal redneckBillusionBiii ultimateBiii revengeBiii remasteredB	ii legacyBii jediB	ii battleBii backBifBicarusBhyrule warriorsBhyper lightB
hunter theBhuntedBhuntdownB	humankindB
human fallBhueBhotshot racingBhotshotBhotelBhorrorBhorizon chaseB	horizon 3Bhoodlum havocBhoodlumBhitman bloodBhitman 3Bhit runBhistory channelBhisBhiredBhighwayBhidden dragonBhidden dangerousBhexBherosBhero aerosmithBhereB	hellpointBhelloBhellblade senuasB	hellbladeBhelixBheavensB
hearts theB	hearts hdBhd theBhd remixBhawx 2Bhat inB	harms wayBharmsBhammersB	halo warsB
halflife 2Bhalfgenie heroB	halfgenieBhadesB
gunslingerBgunmanBguerrilla remarsteredBguacamelee superBguacamelee 2Bgrip combatBgrim fandangoBgrid autosportBgretzky nhlBgretzkyB	greedfallB
great jujuBgreat escapeBgreak memoriesBgreakB
grandia iiBgotham racingBgoreBgooBgoldeneye rogueB	gods partBgoddessBgoatBgladiusBgigaBgiana sistersBgianaBghosts nBghostrunnerBghostly adventuresBghost huntersBget evenBgenesis collectionBgear xxBgauntlet darkBgat outBgatBgardens betweenBgardensB	gangstersBgames tokyoBgame completeBgallop racerBgallopBgalgunB	galactrixBgaiden sigmaBgaiden masterBfzeroBfuserBfuriBfullmetal alchemistB	fullmetalB
full metalBfugitiveBfruitB	frostpunkBfrom russiaB
friends ofBfreedom fightersBfreeBfreddysBfragments ofB	fragmentsBfractured butBfox nBfossilBformulaBforgeBforestsB	ford boldBforce awakensBfor redemptionBfor neighborvilleBfor middleearthB	for honorBfootball 07BfleetBflame inBfive nightsBfitBfinestBfindingBfinchB
final fourBfighting championshipBfighterzB	fighter vBfighter iiiBfighter alphaBfifa 21Bfifa 18Bfifa 14BfezBfencer fBfellowship ofB
fellowshipBfelixBfear effectBfathersBfateextellaBfate 2B
fatal furyBfast furiousBfantasy viiiBfantasy tacticsB
fantasy ixBfandangoBfallen legionBfallen angelB	fall partB	fall flatB	falconeerBfairy fencerBfactorB
faction iiBfablesB	fable iiiBfable iiBf1 raceBeyepetBextremegBextended cutBexplorers ofBexilesBevil villageBeverything orB
everybodysBeverybody runsBeverquest theBeventsBevenB	euro 2008BethanBeternal theBespn nhlBespn internationalBespn collegeBescapists 2Bescape zeroBescape planBescape 2B
erica reedBepisode threeBepBenoughBenigmaBenemy unknownB	ends withBendsBemptyBempires iiiB
empires iiBemblem fatesBelysium theBelusive ageBelusiveBelite forceBelite dangerousBelite 4B	eleven goBeggBedna harveyBednaB	edition 2Bedith finchBedithBedge 2Bearthworm jimB	earthwormB
earthbloodB	earl backBduty infiniteBduty 4Bdungeon defendersBdrumBdropBdrivingB
driver sanBdrillerBdrillBdrifterBdredd vsBdredd dreddBdreamfall chaptersBdreadBdragons chroniclesBdoneBdonaldBdogs legionBdoBdivinity iiBdivineBdiva fBdishonored theBdisciples iiB	disciplesBdiscBdisasterB	dinosaursBdinosaurBdino crisisB
diner dashBdinerBdilemmaBdigitalBdevil survivorB	detentionBdeserveBdeluxe packBdeliveranceB
deliver usBdeliverBdefenders ofBdeclassifiedB
deathmatchBdeath jrBdeadpoolBdeadly allianceB	deadlightBdead nationB
dead cellsBdead byBdc universeB	dauntlessBdarkwoodBdarksiders iiiBdarksiders genesisBdarksideBdarkest dungeonB	dark voidB
dark sparkBdark ofBdark legacyBdark eyeBdark crystalBdark ageBdarius cozmicBdariusBdantesBdaniel negreanuBdanielBdangerous spacetimeBdancingBdance centralBdamacyBdBcursed mummyBcureBcupheadB	cup southBcrystal ageBcrypt ofBcryptBcry hdBcruisnBcruiseBcriterion gameB	criterionB
cris talesBcrisBcrimsonlandBcrime streetsB	crime newBcricketBcreed syndicateBcreatorBcrashersB
crash teamB	crash tagBcrash ofB
crash mindB	crackdownBcozmic collectionBcozmicBcourt tennisBcostume questBcostumeBcosmicBcorpBcopsBcooperBconstructor portalB	conquer 3BconBcomplete sagaBcome deliveranceBcolorBcollege footballBcognition anB	cognitionBcoaster consoleBcoast 2Bclub wastelandBcloverBcloud catcherBclose toBclockBclive barkersBclimbBcivilization ivBcity storiesBcitizens ofBcircusBcircuitBchristie theBchocolate factoryB	chocolateB	chibiroboBchasers nightwarBchasersBchase turboBcharlie andB	chaosbaneBchampionship racingBchampionship dartsBchampions ofBchampionB	chains ofBcellsBcelesteB	celebrityBcause 3Bcatcher chroniclesBcatcherB	castle ofBcastillaB
case filesBcartoonBcarmageddonBcardsBcarBcaptain toadBcaptain americaBcamerons avatarBcamelotBcalamityBcakeBcactusBby daylightB	but wholeBbutBbush rescueBburnerBbunkerBbuildBbugBbrutaleBbrutalBbrotherhood ofB
broken ageBbrideBbrawlersB	brave theBboysBboy andBbowsersB	bounty iiBbounty hunterBbottom rehydratedBboogieBboneBbonBbomberman rBbomberman landBbombB
bold movesBblurBbloody roarBbloodstained ritualBbloodstained curseBblood stoneB
blood omenBblood ofBblood moneyBblizzardB
blitz 2002BbleedsBblazing chromeBblaster masterBblasphemousBblade ofB	blackwoodBblack whiteB	black theBblack knightBbizarre adventureBbizarreBbirthBbirds ofBbioshock theB
bioshock 2BbinaryBbillyBbikesB
beyond twoBbetrayalB
below zeroBbeholderBbee simulatorB	bee movieBbeach volleyballBbattlestationsBbattlefield 4Bbattlefield 1Bbattle royaleBbattle nexusBbattle chasersBbatman vengeanceBbatman riseBbatman 2BbastionBbastardsBbaseball 2k8Bbaseball 2k6BbarnyardBbarkersB
bards taleBbardsBbandicoot theBbandicoot nB	band heroBband 3Bbanana blitzBball ragingBball fighterzBbalanceBbaghdadBbackyard wrestlingBbackyardBaxisBawesomenautsBawayBavalonB	autosportB	auto viceBatv unleashedBathenaB	at portiaB
at freddysB	astro boyBassault androidBary andBaryBartsBarsenalBarmy trilogyBarmadaBarmaBarcaniaBarcade gameBarcade editionB
apprenticeBapocalypse earthbloodBapex legendsB	ao tennisB	ant bullyBannoBannihilationBandroid cactusBandroidBand seekBand monarchsBand hisBancient godsBan ericaB	an empireB
an elusiveBampedBamidBamericasBamerica superBamalur rereckoningBamalur reckoningBalterBallplayBallied assaultB	all starsBaliens versusBalien isolationBalexBakibas tripBagonyBaggressive inlineB
aggressiveBages 3B
afterpartyB	afternoonBafter burnerB	aerosmithBaeroBaer memoriesBaerBadventure continuesBadvance warsBacreBaces ofBaccent coreBaccentBabyss odysseyBabes oddyseeBabc murdersBabcBa wildBa wayBa seriesBa knightBa hatB	a fistfulBa dayBa dangerousBa criterionB	9 monkeysB88B80 daysB80B8 fiaB50th anniversaryB50thB	4 specialB4 arenaB3 titanB	3 specialB	3 nemesisB3 mutantB3 makeB	3 hoodlumB3 dimensionsB3 bloodB2xB2k17B2k16B2k13B2k12B20th anniversaryB20thB2020 theB	2010 fifaB
2006 coastB	2002 fifaB	2 vietnamB2 unleashedB	2 specialB2 silentB2 lostB2 finalB2 featuringB2 fallenB2 directorsB2 dcB2 curseB2 coastB2 bushB2 beyondB2 africaB13th theB101B
09 allplayB007 nightfireB007 everythingB	007 bloodB	zx legacyBzone 2B	zombie inBzombie driverBzombiB
zodiac ageBzodiacBzetaBzeroesBzero zxBzero tvB	zero dawnBzer0 sumBzer0B
zeno clashBzenoBzelda twilightB	zelda theBzelda ocarinaB
zatch bellBzatchBzackB
z ultimateBz sagasBz battleBys memoriesB
your shapeByou lookByou dontByomawariByiik aByiikByetB	yesterdayB
yeah wrathByeahByakuza missionsByakuza 6Byakuza 0Byaiba ninjaBxxl 2B
xx olympicBxrdB	xmen nextBxmen destinyBxiii remakeBxii theBxiaolin showdownBxiaolinBxi sBxgra extremegBxgraBxgames snowboardingBxgamesB
xeodrifterBxenoverse 2Bxenosaga episodeBxenosagaBxcom declassifiedBxbox oneBxbladesBxanaduBx8Bwwe wrestlemaniaBwwe 2kBwulverbladeBwrestling 2Bwreckless theB	wrecklessBwrecker altBwreckerBwrc fiaBwrc 6Bwrc 4Bwrc 3Bwrc 10Bwrath hdBworms revolutionBworms fortsBworldwide soccerBworld toBworld pokerBworld iceborneBworld dxBworld bluesBwords beyondBwordsBwordBwonders planetfallBwonders iiiB
wonderlandBwonderful 101B
wolf chaosB	wizard ofBwitnessBwithin 2B	with fireBwitchesBwitch volumeBwispsBwinter xgamesBwinbackBwinBwillyBwilliams collectionBwilliamsB	wildlandsBwildfireB
wild masksBwii uB
wii sportsBwii fitBwideBwick hexBwickB
who remainB	who needsBwhite witchBwhite nightBwhispersBwhispered worldB	whisperedBwheels unleashedBwheelmanBwheeler americanBwheelerBwhat theB	west taleBwereBweapons overBweaponBwe happyBwe areBwaywardBway outBwatchingBwatch 2B
wastelandsB	was aloneBwarzone earthBwars starfighterBwars squadronsB
wars rogueB
wars retroBwars knightsBwars 3Bwarriors strikeforceBwarriors legendsBwarriors chroniclesB
warriors 9Bwarrior tenB	warrior 3BwarpBwarningBwarmastered editionBwarmasteredBwarlords battlecryBwarhammer iiBwarhammer endBwarcraft iiiBwarbandBwar inBwar 3Bwanted weaponsBwakeboarding unleashedBvvvvvvBvran overkillBvrally 4Bvrally 3Bvoyager eliteBvoyagerBvoyageBvoodooBvolume 1BvoltronBvitaBvisageBvirginiaBvipersB
vinci codeBvikings wolvesBviii remasteredB
vii remakeBvideogame 4BvictoriaBviciousB	via domusBviaBvexxBvesperia definitiveBverdunBver122474487139BverBventureBveniceBveneticaBvelocity 2xBveinB	vegas oldB
vegas deadBvegas 2BvaporumBvanishing ofBvancouver 2010B	vancouverBvalhalla wrathB
va11 hallaBva11Bv theBv groundBup inBup contentsB	up bundleBuntold legendsBuntitled gooseBuntitledBunmechanicalB
unmasked aBunleashed featuringBunleash theBunleashB
universityBuniversalis ivBundisputed 2010B	undertaleB
underminerBundergardenBunbox newbiesBunboxB	unboundedBumbrellaBultramixBultimate racingBultimate muscleBultimate destructionBultimate challengeBuefa championsBtypomanB	typing ofBtype0 hdBtype0B
two towersBtwisted dreamsBtwin mirrorBtwilight princessBtwelve minutesB
tv editionBturtleBturing testBturingB
tunes acmeBtsushimaB
truckers 2BtruckerBtruck championshipBtruberbrookBtroyB	tropico 5B	tropico 4B	tropico 3Btron 20Btrofeo pirelliBtrofeoBtrinityBtrilogy deluxeBtricky towersB	tribes ofBtriangleBtrials fusionBtrials evolutionBtriBtrek voyagerBtrek theBtrek conquestBtrek bridgeBtreasures 3Btreasures 2B
transworldB
transistorBtransformers fallBtransformers devastationBtransferenceBtraitorsB	trails inBtrailblazersBtrading cardBtradeBtrackmania turboBtown ofBtowerfall ascensionB	towerfallBtournament paintballBtournament iiiBtour 12Btour 11Btour 09BtoukidenBtouhou gensoBtouchmasterBtotal warriorBtotal overdoseBtotal destructionBtornado outbreakBtormented fathersBtorment tidesBtorino 2006BtorinoB
top racingBtop gearBtoo deepBtonightBtonelicoBtoneBto winBto rememberBto olliwoodB
to karkandBto hillB	to castleBtitans 2Btitanfall 2B
tiny metalB
tintin theBtintinBtimesplitters futureBtimesplitters 2B	timeshiftBtimes vermintideB	time leftB
time forceBtime exploreBtiger hiddenBtiger 3Bthunder 2004Bthunder 2002Bthrough timeBthrottleB	throne ofB	three theB	those whoBthoseB	thongs ofBthongsB
thomas wasBthomasBthird remasteredBthings 3Bthey areBthemeB
theatre ofBtheatreB
the zodiacBthe xxBthe wonderfulBthe wizardsBthe witnessB	the wispsBthe windBthe williamsBthe willBthe whisperedBthe wellBthe voidBthe uprisingBthe underminerBthe undergardenB
the typingB
the turingBthe townB	the thingBthe testamentBthe templarsBthe technomancerB	the talosB
the systemB
the swordsBthe survivalistsB
the sunkenBthe sumB	the stoneB	the stickBthe stationBthe spiritsB	the spireBthe spectrumBthe sojournBthe sleepingB
the shogunB	the shellB
the searchBthe scorpionB
the savageBthe saboteurBthe robinsonsB
the recordB
the reaperB	the realmBthe quinkanBthe prophecyBthe presequelBthe politicalBthe pinballBthe pillarsBthe pathlessB	the partyBthe pageBthe outsiderB
the orochiB
the orangeBthe occupationBthe necromancerBthe monsterBthe millenniumBthe metatronB
the mediumBthe mastersB
the masterBthe mansionBthe madBthe machinesB	the lionsBthe lineBthe lightningBthe liarBthe lawBthe ladB
the krustyB
the karateB	the ivoryB
the islandBthe invisibleB
the insideBthe iceBthe humankindB	the hordeB
the hiddenB	the heistB	the heartBthe handsomeBthe guardianBthe grimBthe gottliebBthe followingBthe farB
the enigmaB
the druidsB	the dreamBthe dragonsBthe dolphinB
the disneyBthe dinosaursB	the denpaB
the deadlyBthe complexBthe comaBthe colossusBthe clubB
the churchB
the castleBthe cartelsB
the cartelBthe caligulaBthe burningB
the bureauBthe breakoutBthe bradwellBthe blackoutBthe beginningBthe beatlesB	the bandsBthe baconingBthe awesomeBthe assemblyBthe argonautsBthe apesB	the alienBthe 40thBthe 2ndBtharsisBthan aBtetris partyBtetris 2Bterritory quakeBterminator salvationB
tensei iiiBten hammersBtemplarsB	tembo theBtemboBtelltale gameBtelling liesBtellingBtellBtekken 7Btekken 6Bteen titansBteeBtecmoBtechnomancerB
team sonicB
taxi chaosBtatsujinBtastyB
task forceBtaskBtarzanBtaoB
tannenbergB
tango downB
tangled upBtangledB
tamagotchiBtalos principleBtalosBtale innocenceBtale inBtak andBtak 2BtaisenBtails ofBtailsBtaiko noBtacomaBsystem reduxBsynapseB	symphoniaBsydney 2000BsydneyB	swords ofBsword shieldBsword inB
sword cityBswitch forceB
swing golfBsuspectBsurvivor theBsurviving marsB	survivingBsurvivalistsBsurvival instinctBsurvival evolvedBsurge 2Bsurfs upBsurfsBsupreme rulerBsupervillainsB
supersonicB	supernovaBsuperman returnsBsuperhot mindBsuperflyBsuperchargersB
super timeBsuper robotB
super mechBsuper luckysBsuper bustamoveBsunshineBsunsetBsunken kingBsunkenBsunderedBsum ofB	suit zeroBsuicideBsuffering tiesB	substanceBstyx shardsBstyx masterBstuntman ignitionBstudioB
stubbs theBstubbsBstronghold crusaderBstriveBstrikesBstriker gunvoltBstrikeforceB
street volB	street v3Bstreet hoopsBstrangleholdBstranger thingsBstranger ofBstrange brigadeB	strandingBstraight roadsBstraightBstory cyberB	stormriseBstorm 2Bstories theBstonesBstoneflyB
stolen sunBstolen drumsB
still lifeBstick ofBstepBstelaBsteinsgate 0Bsteel battalionBsteamBstarved forBstarvedBstarveB	startopiaBstarsky hutchBstarskyBstarlink battleBstarlinkBstarfleet commandB	starfleetBstar universeBstar heroineBstandBstadiumBsquarepants creatureBsquaredBspyro reignitedBsprint carsBsplatoonB
splashdownBsplashBspiritfarerBspin 4Bspin 2Bspiderman edgeBsphearBspellforce 3Bspellforce 2Bspeedball 2B	speedballBspeed paybackBspeed kingsB
speed heatBspectrum retreatB
spectrobesBspectralBspec opsBspecBspearBspawn armageddonBspartan totalBspartaBspace marineBspace channelBspB
sound mindB	soulstormBsoulcalibur viBsoul suspectBsoul reaverB
soul axiomBsorcererBsonic unleashedBsonic racingB
sonic megaB
sonic boomBsonic advanceBson ofBsomaBsolsticeBsolid hdBsoldnerxBsolar empireBsojournBsoccer slamBsoccer managerBsoccer 2019Bsoccer 2018Bsoccer 2017Bsoccer 2016Bsoccer 2015Bsoccer 2014Bsoccer 2013Bsoccer 2007BsoB	snowblindBsnoopyB
snooker 19BsnookerBsnk vsB
snk arcadeBsnk 40thBsnk 2Bsmugglers runB	smugglersBsmiteB
smash hitsBslugfest 2003Bslug anthologyBslug 3BsleuthBslayer chaosBslaters proBslatersBslam tennisB
slain backBskylar pluxBskylarBskylanders superchargersBskylanders imaginatorsBskullgirls 2ndBskies unknownBskiBskerB	skatebirdBskate adventureBsix theB	six siegeB	six rogueBsisters twistedBsing itBsimulator 17Bsims bustinBsimple storyBsimpleB
simanimalsBsimBsignsBsignalB	siege iiiBside ofBshuffleBshotBshortBshooting starsBshootingBshootBshipBshinobi strikerBshining forceBshiness theBshinessB
shiftlingsBshift 2B	shenmue iBshellshock namBshellshock 2Bshaun murrayBshattered memoriesBsharpBshaq fuBshaqBshapesBshape fitnessBshank 2Bshakedown hawaiiB	shakedownBshadows dieB
shadowkeepB
shadow theBshadow mirrorBshadow heartsBshadow 2Bshaddai ascensionBshaddaiBsetsunaB	set radioBserveBserial cleanerBserialB
separationBsekiro shadowsBsekiroBseiyaBsega soccerBsega gtB	sega bassBseeds ofBseedsBsednaBsecret warsBsecret missionsBsecret armoryBsecond ninjaBsecond encounterBsecond contactBsebastien loebB	sebastienBseason 2000B
search forBsearchBseals fireteamBscribblenauts unmaskedBscribblenauts unlimitedBscribblenauts showdownBscreencheatBscreenBscourgebringerBscorpion kingBscorpionBscoobydoo unmaskedBscoobydoo nightBscoobydoo mysteryBscion ofBscionB
science ofBscholarship editionBscholarshipB
scholar ofBscholarBscapeBscalerBsayonaraBsaviorsB	saves theBsavesBsavage planetBsapphireBsapienzaBsandiego theBsandiegoB
sanctum ofBsan andreasBsam theB	salt lakeBsalt andBsakura warsB	sakuna ofBsakunaBsaint seiyaBsaintBsailsBsagasBsaga continuesBsacred citadelBsacred 2BsaboteurBs echoesBrustBrushn attackBrushnBruseBrunbowBrunawayBrun likeBrulesBrulerBrubyBrubiksBrubble withoutBrubbleBrtype finalBrow gatBroseBroronaBrondoB
rome totalBrome iiBrollingBrollerBrogue warriorBrogue squadronBrogue spearB	rogue opsBrogue corpsBrocksmith 2014BrocketbirdsBrocket arenaBrobotech battlecryBrobot taisenB	robinsonsBroadkillBroad redemptionB
road aheadB	riverbondBriveB
rising sunBrising revengeanceBrising 4Brisen 2BripplesBrings onlineBring ofBrimBride 2B	riddle ofBriddick assaultBricky carmichaelBrickyBrice andBriceBrhythm heavenBrevolution ultramixBrevolution theBrevolution partyBrevolution extremeBrevolution blackBrevivalB	revisitedBrevengeanceBrevenant kingdomBreusBreturns finalBretro evolvedBretoldBresurrectedBresponseBrespectBresource machineBresourceBresogunB	residenceBresetBrescue teamBrerollBrequiredB
republiqueBreplicant ver122474487139B	replicantBrenegade opsBremothered tormentedBremothered brokenBremixedBremember meBremainBreloadBrelictaBreignited trilogyB	reignitedBreichBredemption 2Bredcard 2003BredcardB	red riverBred oneBred johnsonsBrecon wildlandsBrecon futureBrecon breakpointB	recompileBrebelsB	rebellionBrebel withoutBreaverB	realms ofBrealms demonBrealm rebornBready 2BreadyB	read onlyBreadBreBrazors edgeBrazorsBrayman arenaBraw 2011Braw 2007Braven legacyBrangoBrampage totalB	rambo theBramboB	rally evoBrallisport challengeB
rallisportBrainbow moonBrailway empireBrailwayBrailroad tycoonB
raiders ofB
raider theBraidBraging justiceBrage ofBracing syndicateBracing nitrofueledBracing evolutionBracing challengeBracing associationBracing 3Bracers 2Bracer unboundedBrace theBraccoon cityBraccoonBr racingBquinkanBquest iiBqube directorsBquantum conundrumB
quake warsB	quake iiiBquackersBqbertBpuzzle leagueBpuzzle agentBpuyo popB	pure poolBpure farmingBpunishmentsBproving groundBprovingBprototype 2BproteusB	pros 2008BpromBproject zooBproject snowblindBproject originBproject edenBpro truckerB
pro surferBpro raceBprison breakB	principleBprince caspianBprey mooncrashB	presequelB	prejudiceBpowerupBpoweredBpotter collectionBpostmodern rpgB
postmodernBposeidonBportugalBportal knightsBportal 2B	porcelainBpolitical machineB	politicalBpolice paintB	polarizedB
poker tourBpoker nightBpokemon rangerBpoisonBpoint ofB
point fallBplux adventureBpluxBplaystation moveB
playgroundBplayerunknowns battlegroundsBplayerunknownsBplayboy theBplayboyBplay controlBplay baseballB
planetfallBplanet 3Bplanet 2Bplaneswalkers 2014Bplaneswalkers 2012Bplanescape tormentB
planescapeBplague taleB	place youBpix theBpixBpitBpirelliBpirates theBpirates andBpipBpinball fx3B
pinball 3dBpilotBpikachuBpidBphotoBphantom painBphantom breakerBphantom braveB	phantasmaBpetBpesBperfect darkB
perceptionBpenumbraBpenal colonyBpearl harborBpeaky blindersBpeakyBpeachBpayne 3Bpayne 2BpatriotsB	patricianBpathlessB
pathfinderB	past cureB
party hardBpartsBpark operationBparallel linesBparallelBparadise remasteredBpanzersBpanic inBpaint itBpaintBpageBpackageBpacific theaterBpacerBoverride mechB
override 2BoverpassBoverlord iiBoverkill editionB	overdriveB
overdose aBover normandyBoutwardBoutsiderBoutlaws legendsBoutlandBouter wildsBoutcast secondB	othercideBorochi sagaBorochi 2Borigins awakeningBorcs andB
orange boxBops theBops 4Boperencia theB	operenciaBoperation raccoonBoperation genesisBonly memoriesBonline morrowindBonline fatalBonline episodeBonline editionBonline aBonechanbaraB	one livesBone ironBone editionBon darkB	on cloverBon 3BolliwoodBolliolli2 welcomeBolliolliB	old bloodBoilBof zB	of virtueB	of vipersB
of victoryBof tsushimaB	of tintinBof symphoniaBof swordBof stoneB	of sorrowBof slimeBof skerBof sherlockBof seasBof riceBof reckoningBof preyBof pipBof orionBof orcsBof numeneraBof neoBof napishtimBof mythologyBof mystraliaB
of mysteryB	of murderB
of mineralBof medanBof maxB	of legendBof korraB
of justiceBof jujuB	of impactBof himB	of heavenBof gunsB	of gothamBof gooBof gloryB
of generalBof gangstersBof furyBof ethanBof emergencyBof drBof dittoB	of demonsBof deceptionBof deadBof cybertronB
of courageB	of cortexB
of controlB
of celcetaB
of carnageBof brotherhoodBof billyB
of balanceBof azureB	of avalonB	of arkhamBof allB
of agarestBof 100BodeBoddysee newBoddballB	ocean theB
occupationB
ocarina ofBocarinaBobserver systemBoathBnumeneraBnumberBnuclearB	noire theBnocturne hdBno tatsujinBno straightBno oneBno mercyBno goingBnitrofueledB
nitro kartBnintendogs catsBnintendo 3dsBninjagoBninja revolutionBninja heroesBninja councilBninja 2B
nihilumbraB	nights ofBnight inBnight combatB
night callBnight atBnight 2Bnier replicantB	nidhogg 2B	nicktoonsBnickelodeon allstarBnhl 2k9Bnhl 2k6Bnhl 2k3Bnhl 2k10Bnhl 2002Bnhl 2001Bnhl 08Bnfl footballB	nfl feverBnfl 2k2Bnfl 2001Bnfl 13Bnfl 10Bnext dimensionBnewbies adventureBnewbiesBnew playB	new orderBnew nightmareBnew nBnew friendsBnew dayBnever aloneB	network 5Bnest ofBnestB	nes remixBnerdBneogeoBnemoBneighborB	needs youBneedsBnecromunda hiredBnecromancerBncaa gamebreakerBncaa baseballB
nba insideBnba 2k8Bnba 2k6Bnba 2k3Bnba 2k20Bnba 2k19Bnba 2k18Bnba 2k13Bnba 2k10Bnba 08Bnba 07Bnascar racingB	nascar 08B	naruto toBnaruto ninjaBnarnia princeBnarcos riseBnarcosBnapoleonB	napishtimBnano assaultBnanoBnamedBnam 67BnamBnaildBn tastyB	mystraliaBmysticBmystery mayhemBmyst iiiBmxgp theBmx superflyBmusicalBmusic generatorBmuseum battleBmuseum archivesBmurrayBmurdered soulBmurderedB	murder onBmuppetsBmulakaBmtvs celebrityBmtvsB
mtv sportsB	mtv musicBmouseBmotogp 3B
moto racerBmothergunshipB	more thanBmora exBmoons ofBmoonsB	moonlightB	mooncrashBmoon 3dBmonsters incBmonster promBmonster madnessBmonster boyBmonopoly streetsBmonday nightBmonarchs royalBmoleBmodnation racersB	modnationB
modellistaBmlb 09Bmlb 08Bmlb 07Bmissions ofBmission evolvedBmissing linkB
missing jjBmissileBmisadventuresB	mirror ofBmiracle worldB	minutes aBminuteBminoriaBminisBmineral townBmineralBminecraft playstationBminecraft dungeonsBmind controlB	milky wayBmilkyBmilesBmighty switchBmiddleearth iiBmickey mouseBmickey 2BmexicoBmetricoBmeteosBmetatronBmetamorphosisB	metallicaB
metal wolfBmetal furiesB
metal armsBmercyBmercury meltdownBmercenary kingsBmenhirBmen airBmemories retoldBmementoBmegatonBmega collectionBmeet theBmediumBmedanBmechwarrior 4BmechassaultB
mechanicusBmech leagueB	mech cityBme myBmaximaBmax saveBmattersBmatrix pathBmaternational championshipBmaternationalB
mastermindBmaster zeroBmaster thiefBmassacreBmasksBmarvel heroesBmartyrBmarkerBmario makerBmario 64BmarinesBmarch ofB
marc eckosBmarcBmantis burnBmantisB
mans chestBmanorBmania eggstremeB	manhunt 2BmandyB	mandatoryBmana khemiaBman anniversaryBman 9Bman 10BmamodoBmallBmaliceBmajesticBmaidenBmaid ofBmahjongB	magna cumBmagic vBmagic markerBmagic ivBmagic heroesBmagic clashBmages ofB	mafia iiiBmafia definitiveBmadness returnsBmadagascar escapeBmad onesBmad maxBmachines worldBmacfield andBmacfieldBlyokoB
lynch deadBlynch 2BlumoBluminousBlumines remasteredBluftrausersBluchaB
lost wordsBlost viaBlost sphearB
lost lordsBlost empireB
lost emberBlootBlookBlongest journeyB	long roadB	lone wolfB
loeb rallyBloebBloadedBlives foreverBlivelockB	live 2001Blive 09Blittle kingsBlittle infernoBlittle hopeBlittle acreB
little aceBliterature clubB
literatureBlipsBlionsB	lion kingBlineageBlimitB	like hellB	lightyearB
lights bigBlightning returnsBlightning kingdomBlifeless planetBlifelessB
lichtspeerBliberation hdBliarBlets goBletBlemmingsBlego dcBlegend rebornBleeBlechucks revengeBlechucksBleague 20062007Ble mansBlaudeB
late shiftB
last roundB
last placeB	last hopeBlaserBlarry magnaBlanternBland ofB	lake 2002Blair 3dBladB
kuni wrathBkuni iiBkungfuBkrusty krabBkrustyBkrabBkorraBkonamiBkombat xBkombat deceptionBkombat armageddonBkohanBkoBknoxxB	know jackBknights andB	knight toBknight jediBklonoaBkittenBkissBkings storyBkings iiB	kingmakerBking oddballBkinect sportsB
killswitchB	killer isBkidd inBkiddBkid sagaBkickbeatBkholatBkhemiaBkessenBken follettsBkenBkelly slatersBkellyBkaze andBkazeBkarkandB
karate kidBkarateBkangaroo roundBkain defianceBkai theB	justice 2BjupiterBjuniorBjumperB
jump forceBjuiced 2B
juarez theBjuarez boundBjoyBjokerBjohnsons chroniclesBjohnsonsB	john wickBjj macfieldBjjBjet setBjerichoBjedi fallenBjedi academyBjazzBjaws unleashedB	jam fightBjackass theBjack battleB
ivory kingBivoryBiv oblivionB	iv arcadeBit redB	it brightBislesB
islands ofBisland riptideBisland definitiveBisland 2Bis notBis emptyBis deadBironcastB	iron fromBirisB
invizimalsBinvestigationsBinvestigation hardBinvectorBinuyashaB
intro packBintroBinternational trackBintelligenceB	instinctsBinside driveBinsect armageddonBinsectBinsanityBinquisitor martyrB
inquisitorBinnocentBinkBinjustice godsBininjaBinfinity runnerBinfernalBincredibles riseBincidentBin wonderlandBin tooBin soundB	in shadowBin residenceBin rebelB
in projectBin paradiseB
in miracleB
in minutesB	in mexicoB	impostorsBimport nightsBimpact winterBimaginatorsBikarugaBiii nocturneBiii morrowindB	iii exileB
ii scholarBii revenantBii resurrectedBii enhancedB	ii doubleBii deathinitiveBidolBiconoclastsBiceborneB
ice dragonBi dontBhydrophobiaBhutchBhustleBhunter storiesBhunter freedomB
hunted theBhunt showdownBhundred knightBhumankind odysseyBhuman resourceBhulk ultimateBhulk tacticsBhouse dividedB
hot importBhorseBhorizon zeroB	horizon 4B	horizon 2B	hoops 2k7B	hoops 2k6Bhood outlawsBhood defenderBhonor warfighterBhonor risingBhonor heroesBhonor frontlineBhonor europeanBhonor alliedBhonor airborneB	hong kongBhongBhominidBhomefront theB
homecomingBholmes crimesB	hollywoodBhold emBholdBhockey managerBhobBhitz proB	hitz 2003B	hitz 2002Bhitman introB	hitman hdBhitman contractsBhitman absolutionB	hired gunBhimBhill shatteredBhill homecomingBhill 4Bhill 30Bhill 2Bhijo aBhijoBheyB
heros tailBheroinesBheroineB
heroes theBheroes overB
hero worldBhero warriorsB
hero squadB
hero smashBhero onBhero metallicaBhero inBhero 5Bhero 2BhenkBhelp wantedBhells highwayBhello neighborBhellboy theBhellboyB	hellboundB	hell yeahBhell isBhell damnationB
hedgehog 2Bheavens vaultB
heatseekerB
hearts iiiB
headhunterB
hd editionBhawks provingBhawks downhillB	hawk rideBhawaiiBhaunted mansionBhatoful boyfriendBhatofulBhastings tournamentBhastingsB
harvey theBharvey birdmanB
harmony ofBhardwoodBhardlineB
hard resetB
hard placeBhard evidenceB	happy fewBhandsomeBhalo spartanBhalo combatBhalo 3Bhalo 2Bhalla cyberpunkBhallaBhakuokiB
hackgu volBguzzlersBguysBgunslingers taleBgunslingersBguns ofB	guns goreBgunnerBgunman cliveBgungraveBgundam seedBgundam 2BgtrB
gt advanceB
growlanserBgrow upBgroveBground zeroesBground controlB	gromit inBgrisBgrindBgrim adventuresBgreyBgreg hastingsBgregB	green dayB
greed corpBgreedB	great aceBgravity rushBgraveyard keeperB	graveyardBgravelB
grand slamB
grand agesBgrand adventureBgraceBgpBgottlieb collectionBgottliebBgotham cityBgore cannoliB
goose gameB
golden sunB
golden axeBgoing underB
going backBgoin quackersBgoinBgodzilla unleashedBgods triggerB
gods amongBgodfather iiBgoblins resurrectionBgoblin commanderBgoblinBgoat simulatorBgo goBglobal terrorBglobal offensiveB	glitch inBglitchBgladiator swordBgiraffeBgiganticBgiga wreckerBghostbusters sanctumB
getting upBgettingBgeorgeBgensoBgenesis noirBgenesis classicsBgenesis alphaB	generatorBgeneration zeroBgeneral knoxxBgeneBgears 5Bgear xrdBgear surviveBgear striveBgear risingBgate ofBgas guzzlersBgasBgangs ofBgangsBgangBgames aladdinBgamebreakerBgame seriesBgame remasteredB
game partyBgalakzBgalaga legionsBgaiden zBfx3Bfuture tacticsBfuture soldierBfuture perfectBfury unleashedBfuriesBfurBfunkyBfull throttleB	full clipB	full autoBfu aBfrozen synapseBfrom iceB	from dustBfrogger ancientBfrightsBfreestyle streetB
freekstyleBfreedom planetBfreaky flyersBfreakyB	four sonsBfoundB	foul playBfoulBfossil fightersBforwardBfortune paybackBforts underBfortsBfortniteBforgotton anneB	forgottonBforgotten realmsBforgotten cityBforgedBforest definitiveBforegoneBford racingBforce reloadedBforce insectBforce 3Bfor nyBfor helpB	for atlasBfor allBfootball 2k3Bfootball 2005Bfootball 2004Bfootball 2003B	followingBfolletts theBfollettsBflyersBfluxBflowBfloor 2BflockersBflockB	flatout 2Bflashpoint redBflashpoint dragonBflashBfive aBfitness evolvedB
fishermansBfishB	first sinBfirst encounterB	firewatchBfireteam eliteBfireteam bravoBfingerBfinest hourBfinding nemoBfighters collectionBfighter 30thB	fifa 2001Bfifa 20Bfifa 19Bfifa 17Bfifa 16Bfifa 15BfictionBfewBfeudalBfestival ofBferrari challengeB	felix theBfeelingB
federationBfeaturing shaunBfeaturing rickyBfeaturing pgaBfearsBfear 3Bfatal bulletBfat princessBfatBfarming 2018B
far harborBfantasy type0Bfantasy maximaBfantasy harvestBfandango remasteredBfallen orderB	fairytaleB
fairy tailBfaery legendsBfaeryBfaction armageddonBfacebreakerB	fable theBf1 championshipBf1 2021Bf1 2020Bf1 2019Bf1 2018Bf1 2017Bf1 2016Bf1 2015Bf1 2014Bf1 2013Bf1 2012Bf1 2010Bf1 2002Bf1 2001Bextremeg racingBextreme skateB
extractionB	explosionBexplorerBexplore theBexploreB
experimentB
excitebikeBevolveBevo 2Bevil theBevil operationBeveryday shooterBeverydayBevery cornerB	everspaceBeuropean assaultB	euro 2012B	euro 2004B	etherbornBethan carterBeternity theBeternity completeBestherB
essentialsBespn winterBescape deadB
escalationBepisode sixBepisode fourBepisode fiveBengine aquilaBenemy territoryBenemy frontB
enders theB	end timesBemperors tombBemberBem upBelvesBelexB
eleven proBeleven 9Beleven 8Beleven 3BelephantBelements ofB
el shaddaiBel hijoBegoBeggstreme madnessB	eggstremeB	egg maniaBefootball pesB	efootballBeffect sednaBeffect legendaryBeffect andromedaBedition lechucksB
edition dxBedge catalystBeckos gettingBeckosB
echochromeBecco theBeccoBeater 3B	earthlockB
earth 2150B	earned inBearnedBduty warzoneBduty finestB	dustforceBdunwallB
dungeon ofBdungeon explorersBdungeon becauseBduel mastersBducktales remasteredB	ducktalesB	duck goinBducati worldBducatiBdrumsBdruidsBdriver parallelBdriver 3Bdriver 2B	driven toB	driveclubB
drakengardBdrakeBdragoonBdragons trapBdragons crownBdragon risingBdownwellBdownhill jamBdouble impactBdouble dBdoomsdayBdont starveBdont dryBdone runningBdonald duckBdomusBdomainBdolphinBdollBdokidoki universeBdokidokiBdoki literatureB	doki dokiBdokaponBdogs 2Bdog daysBdodgersB
doctor whoBdittoBdissidiaBdisneys tarzanBdisneys meetBdisneys magicalBdisneys extremeBdisneys donaldBdisneypixar toyBdisney universeBdisney ducktalesBdisney classicBdisney afternoonBdisintegrationBdishonored deathBdishonored 2BdisguiseB	disgaea 4Bdisaster reportBdirt showdownBdirt 4Bdirt 3BdillonsBdigimon storyB	die twiceBdg2 defenseBdg2BdexBdevotionBdevils daughterBdevil summonerBdetroitBdestiny theB	destiniesBdesperados iiiBdesert childB	denpa menBdenpaB
denied opsBdeniedBdemons forgeBdemon stoneB
demolitionBdeleteBdegreesBdefinitive collectionBdefendBdefeatBdeception ivBdecepticonsBdeathspank thongsBdeaths doorBdeathinitive editionBdeathinitiveBdeath strandingBdeath ofBdear estherBdearB	deadliestBdead warBdead survivalBdead rabbitBdead overkillB
dead moneyBdead menB	dead hailB	dead godsBdead 400Bdc supervillainsB	dc comicsBdaytona usaBdays toBdaymare 1998BdaymareBday rockB
daxter theBdaughterB
darts 2008BdarqBdarkstarBdarkstalkersBdarksiders warmasteredBdarkside detectiveBdarkness theBdarkness iiB
darkest ofBdark watersBdark theBdark summitBdark sectorB	dark roomB	dark fallBdark athenaB
dark angelBdariusburstBdantes infernoBdangerous golfBdanger 2Bdance 4B
dance 2017B
dance 2016Bdamacy rerollBdakar 18BdakarBcyberpunk bartenderBcyberpunk 2077Bcyber sleuthBcurtainBcursed kingdomBcursed crusadeB	cum laudeBcumBculdceptBcubeBcrysis 3Bcrysis 2Bcry wolfB
cry primalBcry newBcry instinctsBcry 6Bcrouching tigerB	crouchingB
crosswordsB
crossroadsBcrossing soulsB	crosscodeBcritterBcrimes punishmentsBcrew 2BcreekB	creed theBcreed revelationsBcreed odysseyBcreed liberationB	creaturesBcreature inBcreature fromBcreamBcrazy machinesBcrawlBcrash nitroB
cozy groveBcozyB
covert opsBcourseBcourageBcounterstrike globalBcosmic starBcorsa competizioneB
corruptionBcopB	conundrumBcontrol deleteBcontracts 2BcontractBcontra rogueBcontents underBcontentsBconquest ofBconflict vietnamBconflict globalBconflict deniedBconcreteB
conceptionBconariumBconan exilesBcompetizioneBcommandos strikeBcommander unleashBcomics adventureBcomicsBcomesBcombat missionBcombat evolvedBcombat 7BcomaBcolors ultimateBcolorfulBcolonial marinesBcolonialBcollege basketballB
collectorsBcollection theBcoliseumB	cold fearBcoffinBcoffee talkBcodename panzersB	code veinB
code lyokoB	cobra kaiBclustertruckBclub iiB	club 2019Bclover islandB	cloudpunkB
cloudbuiltBcloudberry kingdomB
cloudberryBclose combatB	clockworkBclip editionBclipBcleanerBclassics volBclassic gamesBclank futureBcladunBcivilizations iiiBcivilizations iiBcivilization vBcivilization revolutionBcivilization iiiB	city lifeBcity impostorsB
city brawlBcitizenB	cities xlB	circle ofB	church inBchurchB	chrysalisBchronos beforeBchrono phantasmaBchronicles theBchronicles russiaBchronicles remasteredBchronicles indiaBchronicles chinaBchronicles 4Bchronicles 2B	chromagunB	christmasBchoplifter hdB
choplifterBchocobosB
chivalry 2BchinaBchime sharpBchicken policeB	chapter 5B	chapter 3Bchaos xdBchaos bleedsB	channel 5Bchampionship seasonBchampions leagueBchallenge trofeoBchainBcent bulletproofBcell blacklistBcelebrity deathmatchBcelcetaB
cel damageBcelB
cave storyBcause 4Bcause 2Bcatch aBcatalystB	cataclysmBcastlevania harmonyBcastlevania anniversaryBcastle wolfensteinBcastle crashersBcaspianBcase ofBcartoon networkBcartoBcarterBcartelsBcars maternationalBcarrionB
carnivoresB
carmichaelBcarmen sandiegoBcarmenBcaribbean deadB	card gameBcaptain spiritBcapcom infiniteBcapcom beatBcapcom arcadeBcannoliB	candlemanBcanBcameraBcaligula effectBcaligulaBcalamity triggerBcageBcafeBcabelasB	buzz quizBbuzz lightyearBbuzz juniorB
bustin outBbustinBbustBbusBburnout revengeB	burnout 2Bburn racingBbureau xcomBbureauBbully scholarshipBbulletstorm fullBbullets episodeBbulletproofB
builders 2BbugsnaxBbuddyBbubsyBbrutal legendBbroken steelBbroken porcelainBbrinkBbright lightsBbreed 3B
breakpointBbreakoutBbreakersBbreaker battleB	break theBbreadBbravelyBbrain boostB	brain ageBbraidBbradwell conspiracyBbradwellBboxboyB
box officeBbowl 2Bbound inBboulderBboruto shinobiBborderlands aBboostBbon appetitBbomber crewBbomberBbobBbmx xxxB	blues andBblue dragonBblowoutB
bloodrootsBbloodrayne 2B
bloodlinesBblood trailsBblood driveB	block andB	blitz proBblitz hdB
blitz 2003Bblinders mastermindBblindersBbleed 2Bblazblue chronoBblazblue calamityBblastersB
bladestormBblade warbandBblade kittenBblackwood crossingBblacksad underBblacksadBblackout clubB	blacklistBblacklight tangoBblackgate deluxeBblack fridayBbirds trilogyBbirdman attorneyBbirdmanB	biomutantBbinary domainBbilly mandyBbillionsB
big screenBbig redBbig gameBbfg editionBbfgBbeyond lightBbeyond eyesBbeyond blueBberlinBbeowulf theBbeowulfBbeneathBbell mamodoBbellBbejeweled 3Bbeijing 2008BbeijingBbecomeB	because iBbecauseB	beautifulBbeatles rockBbeatlesBbeat emB	beat downBbeastsBbayonetta 2Bbattlefield vBbattlefield hardlineBbattlefield 1942B
battlebornBbattle throughB
battle losBbattle groundsBbattle engineBbattle brawlersBbatman beginsBbass fishingBbasketball 2k3Bbaseball mogulB
baseball 3Bbartender actionB	bartenderBbarkers jerichoB	barbarianBbangaioBbandsBband ofBband 2Bbanana maniaBball adventureBbakugan battleB	baja edgeBbajaBbadass elephantBbaconingBbackbreakerB	back fromBback 4BbabelBbabaBazure strikerBaxeBawesome adventuresBawesomeBawakeBavicii invectorBaviciiBaven colonyBavenB
automatronBautobotsBauto sanBauto modellistaBauto 2Battorney chroniclesBattorney atBattacksBatelier roronaBatelier irisBatari anniversaryBat lawBasylumBassociationBassembly requiredBashes ofBasherons callBasheronsBashBasgardBascension ofBaround everyBarmy 4B	arms roadB
arms hellsBarms glitchBarms earnedB	armory ofBarmoryBarmorBarmikrogBarmelloBarkham asylumBark survivalBark ofBarise aB	argonautsBarena footballBare billionsBarchives volB	architectBarcana heartBarcade spiritsBarcade hitsBarc theBarc ofBar tonelicoBaquilaBaquanoxBapprentice ofBappetitBapesB
apache airBapacheBanthology manBanthology littleBanthemBanswerBanother worldBanomaly warzoneBanneBangels 2B	andromedaBandreasBand sanctuaryBand sacrificeBand ruinBand mirrorsBand menBand bulletsB	and bikesB
and beyondBand aBancient shadowBancestors theBanarchy onlineB	among theBamid theBamericas armyBamerican proBamerican idolBamerican fugitiveB
am setsunaBam breadBam aliveBaltBalpha protocolB	alpha oneBalpha 3Ballstar brawlBalliesB	all fearsBalive xtremeBalive 6Baliens fireteamBaliens colonialBalien hominidBalice madnessBaliasB	alex kiddBaladdin andB
air combatBair assaultBaiBaheadB	agents ofBagent underBagendaBage definitiveBage dawnBagarest warBagarestBafternoon collectionBafro samuraiBafroBafrikaBadventure palsBadventure onBadventure 2B	advance 3B	advance 2BadB
activisionBactive 2Baction henkBact ofBacrossBacme arsenalBacmeBace inBacademiaBabzuBabsolverB
absolutionBaboveBaaeroBa worldBa solarBa simpleBa rideBa realmBa pulseBa postmodernBa plagueBa nestBa masterBa legendB	a kingdomBa houseBa herosBa hardBa gunslingersB	a feelingB	a fantasyBa dcBa causeBa blockB99B98B8 prejudiceB8 internationalB	8 empiresB7 skiesB67B6 worldB6 theB
5 ultimateB
5 strikersB	5 specialB5 polarizedB5 lostB5 lastB5 cryB5 cityB4x4 evolutionB4x4 evoB40th dayB40th anniversaryB40000 spaceB40000 mechanicusB40000 inquisitorB400 daysB400B4 whoB
4 ultimateB4 roadB
4 guardianB4 getB4 fiaB4 farB4 escapeB4 darkB4 bloodB4 automatronB4 aroundB4 amidB3dsB30th anniversaryB30thB3 wastelandsB3 riseB	3 ripplesB3 razorsB3 petsB3 operationB3 onB3 nightB3 newB3 moreB3 longB3 inB3 hellB3 drivenB	3 descentB3 bfgB3 beyondB3 backB3 aB
2nd encoreB2k15B2k1B2k battlegroundsB2darkB25 toB24B2150B2077B	2064 readB2064B2019 featuringB2014 editionB2010 theB2008 theB20062007B2006 theB2004 portugalB
2 warlordsB2 tiesB2 superB2 substanceB	2 starvedB2 smokeB2 shadowkeepB2 secretB
2 sapienzaB2 rumbleB2 rulesB2 rubbleB2 pointB2 outB2 offB2 modernB2 menB2 marvelB2 hotB2 hideB2 hdB2 grandB
2 forsakenB2 dogB
2 childrenB
2 castawayB2 bloodB2 bigB
2 assemblyB	2 arrivalB2 armageddonB1998B198xB1979 revolutionB1979B1943B
18 wheelerB1111 memoriesB1111B10thB100th editionB100thB100 frightsB10 theB	10 secondB1 zer0B1 tiesB	1 tangledB1 roadsB1 realmB1 penalB1 inB1 heroB1 faithB1 doneB1 chrysalisB1 awakeB08 theB07 theB04B	007 agentB0 hdBzwei theBzweiBzumba fitnessBzumbaBzumas revengeBzumasBzoocubeBzonesBzombies mustBzombie islandBzoidsBzodiarcsBzherosB	zero hourBzero directorsBzero 2Bzerker xBzerkerBzelda skywardBzelda majorasB
zelda fourBzelda aBzathuraB	zack zeroBz theBz supersonicBz shinBz burstByugioh worldByugioh legacyByu yuB
yu hakushoBys viiiBys sevenBys ixByours isB
your graceByour friendsByour creationsByou toByou canByou areByoshis woollyByoshis islandByoshiB	yoostar 2ByoostarByoByggdra unionByggdraByetisByet itByesterday originsByes yourByesByellowB	years warB	year walkByarnByakuza remasteredByakuza 5Byakuza 4Byakuza 3B	yaiba theByagerBxxl 3B
xv episodeBxuanyuan swordBxuanyuanBxtreme 2BxoticBxonicBxmen mutantBxiv stormbloodBxiv shadowbringersBxiv heavenswardBxilliaBxiii2B	xi chainsBxgiii extremeBxgiiiBxfilesBxenon racerBxenonBxenoBxbox 360Bx3Bx zoneB	x machinaBx labyrinthBx hdB	x commandBx collectionBwwii theBwwii paratroopersBwwf smackdownBwwe legendsBwwe dayB	wwe crushBwwe 2k20Bwwe 2k19Bwwe 2k18Bwwe 2k17Bwwe 2k16Bwwe 2k14Bwwe 13Bwwe 12Bwta tourBwtaBwrong numberBwrong dimensionBwrestling dontBwrestlemania x8Bwrc 7Bwrc 5Bwrc 2Bwrath unleashedBworms worldBworms blastBworms battlegroundsBworms battleBworms 4Bworms 2Bworlds perilBworlds murderBworlds apartBworld treesBworld seekerBworld remasteredBworld partyB
world nextBworld grandB
world goneB
world golfBworld circuitBworld brothersBworld beneathBworkshopBworkB
word coachBwoolly worldBwoollyBwoolies strikeBwooliesBwoodB
wonders ofBwonderful lifeB
wonderbookBwomensBwolffs gravityBwolffsBwolfenstein 3dBwoe andBwoeBwizardry labyrinthBwithout warningB	with yourBwith aB	witcher 2B
wipeout hdBwinterbottomB	winter ofBwing commanderBwinds ofBwindowsBwindjammersB
wind wakerB	winback 2BwillowsBwill beBwildlifeBwild thingsBwild theBwild heartsB
wild earthBwii playB	wii partyBwii editionB	wide openBwhyBwhos watchingB	whos thatBwho theB
white wolfBwhite versionBwhite skateboardingBwhite marchBwhite knightBwhite 2Bwhispers ofBwhispering willowsB
whisperingB
whirl tourBwhirlBwhiplashBwhipBwhen theBwheels stuntB	wheels ofBwhats yoursBwhat weBwhat didB	what ailsBwest pigsysB
werewolvesB
wererabbitB	were noneBweapon atomicBweakest linkBweakestBwe theBwe skiBwe loveB
we deserveBwe cheerBway tripBwaverlyB	wave raceBwattamBwater tastesBwatching whoBwatch blastersBwastelandersB
was aroundBwarshipsBwarship gunnerBwarshipBwarsawBwars soBwars lethalBwars galacticBwars empireBwars demolitionBwars bountyBwarriors taleBwarrior kingsBwarpathBwarmindBwarlordBwarlocksB
warlock ofBwarioware incBwarioware diyB
wario landBwarhammer vermintideBwarhammer markBwarhammer battleBwarcraft theB	war worldBwar ultimateBwar thunderB
war shogunBwar sagaB	war earthBwar collectionB	war chestBwar assaultBwar 4B
wandersongBwanderer theBwalkBwakes americanBwakesBwakerBwake theBwake remasteredBwacky racesBvs severBvs jackBvs greyBvs dcBvr caseBvr aBvoodoo vinceBvonB
volume oneB	volume iiBvoltron defenderBvoid bastardsBvisionBvirusBvirtues lastBvirtuesB	virtualonBvirtual rickalityBvirtua questBviolenceBvinci disappearanceBvinceBviking battleBviii lacrimosaBviii journeyBvideogame 3Bvideogame 2Bvictorious boxersB
victoriousBvictoria iiBvictisBvicious sistersBvfrBveryB	version 2Bvermintide 2Bvergils downfallBvergilsBverge 2Bverdict dayBverdictBvelvet assassinBvelvetBvegas lonesomeBvegas honestBvault ofBvasara collectionBvasaraBvarnirBvanishing pointBvandal heartsBvandalB	van halenBvampire rainBvambrace coldBvambraceB	valley ofBvalkyria revolutionBvalhalla theBvaccineB
v3 killingBv burstBuzumaki chroniclesBuzumakiButawarerumono maskBus backBuruBurban chaosBup yourB	up exceedB
untold theBunto theBuntoB
until dawnBuntilBunsigned legacyBunsignedB	unrivaledB	unreal iiBunreal championshipBunmechanical extendedBunlimited cruiseBuniverse onlineBuniverse atBuniversalis romeBuniversalis iiiBunity ofB
unity deadBunicornBunhingedB
unfinishedBunepicBunemployed ninjaB
unemployedB	undressedBundisputed 3Bundisputed 2009Bunderhive warsB	underhiveBundead undressedBundead nightmareBuncharted theBunchained bladesBuncanny valleyBuncannyBunbound worldsBunbound sagaBunborn starBunbornBumbrella corpsBumbral starBumbralBultra despairB	ultra ageBultimaxBultimate tenkaichiBultimate showdownBultimate puzzleBultimate mortalBultimate knockoutBultimate hdBultimate genesisBultimate fitnessBultimate evilBultimate carnageBultimate bandBuglyBufc personalBufc 4Bufc 3Bufc 2B	u editionBtyson heavyweightBtyson boxingBtyrannyBtypoman revisedBtyphoonB	tycoon iiBtwonkiesB
two realmsB
twinsanityBtwilight ofBturtles mutantsBturrican flashbackBturricanBturok 2BturnonB
turnip boyBturnipB	turismo 5BturfBtunnelBtunguskaBtunes spaceB
tunes backBtumbleBtsunamiBtsumBtsubasa riseBtsubasaBtry thisBtryBtruck simulatorBtrozeiBtrover savesBtroverBtropical freezeBtropicalBtrophyBtriple packBtrip undeadBtrip hellboundB
trilogy hdBtrigger witchBtrigger happyBtribulationsB
trials andBtrek tacticalBtrek starfleetBtrek shatteredBtrek legacyBtrek dacB	trees woeBtreesBtreeBtraxBtravis strikesBtravisB	travelersB	travel inBtravelB	trap teamBtransworld surfBtransport feverB	transportBtraitors gateBtrainzBtraining inBtrainer theBtrain simulatorBtraderBtrackmania 2Btrack racingB
track packBtrack challengeBtraceBtoysBtownsBtowering adventureBtoweringBtournament ofBtournament 2004Btour tennisB	tour 2002B	tour 2001Btour 14Btour 13Btouhou kobutoBtouch detectiveB
total teamBtotal insanityBtotal immersionBtotal annihilationBtormented soulsBtorment enhancedBtorgues campaignBtorguesBtomorrow comesBtokyo xanaduBtokyo mirageBtokyo legacyBtokyo 42BtokobotBtohuBtoemBtodayBtoby theBtobyBto wrestlemaniaB
to silenceBto runBto powerB
to ostagarB
to freedomBto fightB
to explainB
to deserveBto deathB
to daytonaB	to canadaBto anB
tna impactBtnaBtmnt mutantBtjBtitan soulsBtipping starsBtippingBtinker cityBtinkerBtimespinnerBtime toBtime piratesBtime machineBtime andBtime 3dBtim burtonsBtimBtillB	tiger theBtiberium warsBtiberiumBthunder wolvesBthunder forceB	throwdownBthrottle remasteredB	thrones aBthree housesBthree fourthsBthis myBthis atBthird strikeB
things areB
thieves inB	thief theBthief deadlyBthicker thanBthickerB
there wereBthere isB
there goesB
then thereBthenB
theme parkBthemBtheatrhythm finalBtheatrhythmBthea theBtheaB	the yetisB
the yellowB
the xfilesB
the worldsBthe wooliesBthe woodBthe werewolfBthe wererabbitBthe weakestB	the waterBthe warlockBthe vrB	the vaultBthe unemployedB
the umbralBthe twonkiesBthe twilightBthe travelerBthe terminatorBthe tentacleB	the talesBthe takeoverB	the takenBthe swindleBthe swapperBthe suicideB	the storyBthe spongebobB
the sphinxBthe somniumB	the solusBthe sojournerB	the snailBthe smithsonianBthe sixstringB
the silentB
the signalB
the shieldBthe shapeshiftingBthe secretsBthe screamingB	the scionBthe sandB	the royalB	the rosesB	the romanBthe roleplayingBthe rioB
the ringedB
the rescueB	the remixB	the reichBthe raptureBthe raidersB	the questBthe promisedB	the polarBthe pittB
the pirateBthe persistenceB	the penalBthe pedestrianBthe passingBthe painBthe pactBthe owlsB	the otherBthe operativeB	the omegaBthe oathBthe nineBthe nileBthe neighborhoodB
the museumBthe muppetsBthe multiverseBthe moonlightBthe misadventuresBthe mindgateBthe millionairesB
the mightyBthe metronomiconBthe manhuntersBthe mainBthe maidB	the magesBthe longingB
the londonB	the lightB
the legacyBthe knightsBthe kiwiB	the kingsB
the jungleBthe invisiblesBthe inkBthe immortalsBthe hongBthe hinokamiBthe heroB	the hardyBthe guyB	the guildBthe goodBthe godsBthe getawayBthe generalB	the gatesBthe gallowsBthe furiousB
the frozenBthe freeBthe fowlBthe foundationBthe forbiddenB	the fightBthe fateboundsBthe fastB	the fancyB
the familyBthe ezioBthe expendablesB
the exiledBthe endlessBthe enchiridionB	the dukesBthe duelistBthe dragonflyBthe dishwasherBthe dimensionalBthe diceB
the depthsB	the demonBthe darkestBthe cupBthe crimsonBthe criminalB	the countBthe coreB	the climbB	the chimpBthe childrenB	the chaseB	the chaosBthe cameronBthe bumblebeesBthe brigmoreB
the bridgeB
the breachB	the braveB
the bourneBthe boldB	the bloodB
the blightBthe blackwellB	the beastBthe battlefieldBthe awakenedB
the ascentB
the artfulBthe arrivalB
the arcadeBthe apprenticeB	the angelB
the amuletB
the almostBthe alchemistsBthe akkadianBthat remainsBthat flyingB
than waterB
texas holdBtexas cheatBtetrobot andBtetrobotBtetris ultimateBterranBterminator dawnBtentacle remasteredBtentacleBtensei strangeB	tensei ivBtensei digitalB
tennis proBtennis mastersBtenkaichi 3Btenkaichi 2Btenchu shadowBtemplars theBtemplarBtempest 4000Btelltale definitiveBtell meBtelevisionsBtekken 5B
tecmo bowlB
technologyBtears ofBtearawayBteam xB
team sabreBteam frenzyBteam controlB
taz wantedBtazBtax evasionBtaxBtatsujin drumBtastes likeBtastesBtarzan untamedBtargetBtappingoBtapoutB	tank tankBtangBtamriel unlimitedBtamrielBtamagotchi connectionBtalking andBtalkingBtakoBtakeoverB
taken kingBtake usBtake onBtaisen originalBtaintedB	tag forceB
tag battleBtactics ogreBtactics bladesB	tactics 2Btactical assaultBtachyon projectBtachyonBtable tennisBsystem shockBsynchronicity tomorrowBsynchronicityBsymmetryB
syberia iiB	syberia 3Bswordcraft storyB
swordcraftBsword shadowBsword 7B
switchballBswindleBsweetBswat globalBswat 4Bswashbucklers blueBswashbucklersBswapperB
survivor 2Bsurvival ofBsurfingBsupremacy mmaBsuperstar sagaBsupersonic warriorsB
superpowerBsuperman shadowBsuperhot vrBsupercross worldBsupercross 2000Bsupercar streetBsupercarBsuperbeat xonicB	superbeatBsuper swingBsuper neptuniaBsuper mutantBsuper hydorahBsuper dungeonBsuper dragonBsuper darylBsuper cloudbuiltB
super caneBsunset overdriveBsunlightBsunlessBsun andB
summoner 2B	summersetBsummer memoriesB	summer inBsummer athleticsB
suicide ofB
suffer theBsufferB
suelle theBsuelleBsuburbiaB	submergedBsublevel zeroBsublevelBsubjectBsubBstyle savvyBsturmovik birdsBstuntsBstunt trackBstuart littleBstuartBstruggleBstrong badiaBstrings clubBstringsBstrikes againBstrike vectorBstrike teamBstrike onlineBstrike fightersBstrike backBstrike 4B
streetwiseB
streetballBstreet soccerBstreet homecourtBstreet challengeBstrange journeyBstrandedBstrainBstory 2B
stormbloodBstorm revolutionBstorm generationsBstories untoldB	stories 2Bstop believinBstolen memoryBstoleBstokedBstikboldB	steredennB	step fromBstellaris consoleBstellaBsteinsgate eliteB
steel ratsBsteel ivB	steel iiiBsteel iiBsteel divisionBsteel diverBsteambot chroniclesBsteambotBsteam editionBstealth forceBstealth assassinsBstealBstartingB	start theBstartBstarship troopersB
starlancerBstardust ultraB	stardroneB	stardriveBstar varnirBstar portableBstar commandBstand aloneB
stalingradBstairway toBstairwayBstageBsquigglepantsBsquids odysseyBsquidsBsquarepants movieBsquad assaultB	spyro theBspyro enterBsprint vectorBsports skateboardingB
sports mmaBsports championsBsports 2002Bsports 2BsportB
spore heroBspongebob squigglepantsBsplosion manBsplosionBsplatterhouseB
splatoon 2Bsplat renegadeBsplatBsplasherBspiral horusBspiralBspintires mudrunnerBspinner cayBspinnerBspikeBspiderman milesB	sphere ofB
spelunky 2B	spelunkerBspellsBspeed racerBspeed nitroBspectral forceBspectraBspecterBspeciesBspecial gigsBspartan assaultBsparrowBsparkle unleashedB	sparkle 2Bspare partsBspareBspadesBspacebase startopiaB	spacebaseB	space runB
space raceBspace junkiesBspace ignitionBspace extractionBspace empiresB
space crewBspace colonyBspace assaultBsourceBsourBsound shapesBsoulcalibur vBsoulcalibur ivBsoul sacrificeBsorryBsorrowsBsorcerer kingBsonics ultimateBsonicsB
sonic rushBsonic rivalsB
sonic lostBsonic cdB	sonic andBsongbringerBsonataBsomnium filesBsomniumBsolus projectBsolusB	solseraphBsoloB	solitaireBsolid portableBsolid 3B
soldnerx 2Bsoldiers warBsoldiers iiBsolB	sojournerBso longBsnowboarding 2002BsnowboarderB	snowboardB	snoopy vsBsnocross championshipBsnocrossBsnk heroinesBsnipperclipsBsneakBsnapBsnake eaterBsnailBsmithsonianBsmashing driveBsmashingB
smash packB
smash carsBsmartBsmall radiosBslugfest loadedBslugfest 2006Bslug xxBslug 4Bsludge lifeBsludgeBslimesanBslender theBslenderBsleeping dragonBslayersBslayer kimetsuBskyward swordBskywardB	skyrim vrBskyrim specialBskyrim dawnguardBskylanders trapB	skyheroesBskyeBskydriftBsky soldierBskelterB	skater xlB	skater hdBskater 5Bskate itB
skate cityBskate 3Bskate 2Bsk8landBsizzle serveBsizzleBsize mattersBsizeB	sixstringBsirenBsir hammerlocksBsinnersBsingstar queenBsingstar popBsingstar abbaBsin enhancedBsimulator xBsimulator 2013Bsimulator 2004Bsimulator 19Bsimulator 15Bsims medievalBsimpsons arcadeBsimcity creatorB	simcity 4B	sim worldBsilver starBsiliconB
silhouetteB
silent bobBsilence theBsigns ofBsignBsigma 2Bsiege iiBshuBshrek foreverBshredBshowtimeBshowdown legendsBshow 21Bshots tennisB
short hikeBshoreBshootout 2001Bshogun totalBshogun 2Bshivering islesB	shiveringBshipsBshippuden ninjaBship simulatorBshinovi versusBshinoviBshining soulBshining resonanceBshineBshin budokaiBshiftyBshift iiBshenmue iiiB
shenmue iiBshell standBsheeps clothingBsheepsBsheepBshe rememberedBsheBshaun palmersBshattered universeBshattered unionBshattered skiesBshatterBshapeshifting detectiveBshapeshiftingBshape ofBshaolin monksBshantae riskysBshallieB
shady partBshadyBshadwenBshadows awakeningBshadowlandsBshadowgroundsBshadowbringersBshadow warsBshadow tacticsB
shadow opsB
shadow manBshadow fallBshadow editionBshadow dragonBshadow complexBshadow brokerBshadow assassinsBseverBseven starsBseven sorrowsBsessions feBseries arcadeB	sentinelsBsentinelBsenko noBsenkoBsengoku basaraBsemispheresB	semblanceB	selectionB
sega smashBseekerBsecret worldBsecret mineBsecret hideoutBsecret agentBseasons afterB
season oneB	season ofBseason 2Bseal arbitersB	sd gundamBsdBscreaming narwhalB	screamingBscreamBscrats nuttyBscratsB	scraplandBscrapBscrambleBscourgeBschrodingers catBschrodingersBschool musicalB	scarygirlBscarlett andBscarlettBsbk superbikeBsayonara wildBsay noBsayBsaw iiBsavvyBsavioursBsaviors returnBsave meB
sarges warBsandB	sanctum 2Bsamurai heroesB	samurai 4B	samurai 3B	samurai 2BsamuelB	sammunmakBsamba deBsambaBsam nextBsam iiB
sam doubleBsam 3Bsairento vrBsairentoBsaints sinnersBsail iiBsailBsaga ofBsaga frontierBsafecracker theBsafecrackerBsacred 3B	sackboy aBsackboyBsableBsaberB	ryza everBryse sonBryseB	rygar theBrygarBrustlerBrust consoleBrush forBrush aBrunnersBrunner3BrunersiaB	runaway aBrumble rosesBrumble arenaBrulzBruining blueBruiningBrugby worldB
rugby 2005B
rugby 2004Brugby 20Brubiks worldBrubBrtype dimensionsBrtsBroyale 4Broyale 3B
roundaboutBround 4Brory mcilroyBroryBrorona plusBroot ofB	root filmB	rooms theBroomsBroom vrBrondo ofBrondeBroman empireBromanBrolling westernBroleplaying wargameBroleplayingBrogue remasteredB
rogue acesB	rodea theBrodeaBrocky legendsBrockstar gamesBrockstarBrocksBrocketmen axisB	rocketmenBrocketbirds hardboiledBrocket knightBrock revolutionBrock casinoBrochardBrobotsBrobotoBrobotech invasionBrobot revolutionBrobocalypseB	roboblitzBrobert ludlumsBrobertB	road tripBroad notBroad 96Brlh runBrlhBriviera theBrivieraB
river kingBrival swordsBrival megagunBriskys revengeBriskysBrisk ofBrising tripleBrising tideB
rising theBrising stormBrising aB
rise shineBripd theBripdBriot responseBrio 2016Brings aragornsBringed cityBringedBrights retributionB	rights iiBridingBrider 2Bride orBride 3Briddick escapeB	rickalityBrick andBrickBribbit kingBribbitBrhemBrez infiniteBrewardBrevolverBrevolution volumeBrevolution universeBrevolution supernovaBrevolution presentsBrevolution hottestBrevolution disneyBrevisedB	revenantsBrestaurant empireB
restaurantBresonance refrainBresonance ofBreset reduxBrequestBrepublique episodeBreport 4BreplayBrengokuBrenegade paintballBrenaissanceBremnant remasteredBremembered caterpillarsB
rememberedBremastered editionBremake intergradeBremains obscureBreincarnationBregenerationBrefrain covenBredout spaceBredemption undeadBredemption liarsBredeemer enhancedBred sunBred stringsB	red steelBred starBred solsticeBred orchestraB	red ninjaBred mercuryBred hotBred goddessB
red deluxeBred bullB	red baronBrecutBrecoreBrecon islandBrecodeBrebootedBrebel galaxyBreaver 2B	rearmed 2B
real worldBreal warB
rc revengeBraystormBrayman 2BrayBraw 2006BravensBraven remasteredBraveBratBrascalsBraptureBrapstarBrapalaBransomBrally fusionBrally 3B
rally 2005Braji anBrajiB
rain worldBraidou kuzunohaBraidouBraider definitiveBraider chroniclesBraiden vB
raid worldBragnarok odysseyB
rage burstBrag dollBragB
radios bigBradiosBradicalBradiant historiaBradianceBracquet sportsBracquetBracing worldBracing technologyBracing legendsBracing championshipBracing 4Brachel fosterBrachelBraces crashBracesBracer driftBracer 3B	raceoramaBrace ofBrabbids travelBrabbids kingdomB
rabbids goB	rabbids 2BquotientB
quiz worldB
quietus ofBquietusBquietB
quest viiiB
quest packBquest monstersBquartersBquarterback clubBquarterbackBquarkBquantum theoryBquantum breakBquake 4Bqbert rebootedBqb clubBqbBpyreBpuzzlingBpuzzlegeddonBpuzzle kingdomsBpuzzle dragonsBpuzzle dimensionBpuzzle chroniclesBpushmoBpushBpursuit unhingedB
pursuit ofBpursuit forceBpurpleBpure futbolB	puppeteerB
punishmentB	punch manBpump itBpumpBpuchiBpsychopass mandatoryB
psychopassBpsychokinetic warsBpsychokineticBpsycho circusB
psiops theBpsiopsBpsikyo shootingBpsikyoBpsBprophecy ofBpromised landBpromisedBpromiseB	promathiaBprologueB
projectiveB	project xBproject rootBproject poseidonBprofessor heinzBprofessionalBpro tournamentBpro snowboarderBpro fishingBprison architectBprincess andBprincesBprimetime 2002B	primetimeBprimal carnageBprideBpresents tableBpresents americanBpremonition theBpremier editionBpremierBpreludeBprehistoricBpredator huntingBpredator extinctionBpredator concreteB
predator 2BpraetoriansBpractical intelligenceB	practicalBpowersB
powered upBpower tennisBpower stoneB	power gigBpower battlesBpostapocalypticBpostal 2BpostBpossibleBportable opsBpopolocroisB	pop feverBpool paradiseBpool nationBponyBpongBpolice 2BpolarisBpolar panicBpolar expressBpokken tournamentBpokkenBpoker featuringBpokeparkBpokemon whiteBpokemon ultraBpokemon snapBpokemon letsBpokemon blackBpokemon battleBpoison controlBpoint lookoutBpodeBpneuma breathBpneumaBplus alchemistsBpleaseBplaystation vitaBplaystation allstarsBplaystation 4Bplaygrounds 2B	play 2002BplatoonBplatinumBplanetside 2Bplanet premierBplanet battleB	planet 51Bplaneswalkers 2013BplanesB
plan bravoBpk outBpkBpixeljunk edenBpixel storyBpixel piracyBpittB
pit peopleBpistol whipBpistolBpirates carnivalBpirates bootyB
pirate godBpiracyBpioneersB	pinstripeBpingBpinball arcadeBpimp myBpimpB
pilotwingsBpikunikuBpikmin 3Bpigsys perfectBpigsysBpigsBpiece worldBpiece piratesBpiece burningB
picross 3dBphoneBphoenix pointBphenomBphaseBpharaohsBphantom doctrineBphantasma extendBpes 2020B	persona 2BpersistenceBpersia rivalBpersia classicB	perimeterBperil onBperilB
perfect 10BperBpenariumB
penal zoneB	pen paperBpenBpegasusBpedroB
pedestrianBpb winterbottomBpbB
payday theBpawarumiBpatrolBpathwayBpathsB
pathologicBpathfinder kingmakerB	patapon 2Bpast wasB	past fateBpassingBparty deluxeBparty 2BpartnersBpart ofBpart iiBparisBpariahB
parchmentsBparatroopersBparasite eveBparasiteBparadise lostBparadise killerBparadise islandBparadeBpapo yoBpapoBpanzers phaseBpanzer tacticsBpanzer paladinBpanzer generalBpanzer dragoonBpanzer corpsBpants adventuresBpang adventuresBpangBpanda 2Bpalmers proBpalmersBpaladinB	paintingsBpaintball maxdBpaintball 2009BpactBpacman partyBpacman museumBpacman feverB
pacman 256Bpack 7Bpack 5Bpack 1Bpacific stormBownBowls ofBowlsBovertureBoverlandBovercooked allBoverclockedB	overboardBoutlaws sprintBoutlaw volleyballBoutlaw tennisBoutdoorBout togetherBourBotogiBother watersB
other sideBostagarBorigins theBorigins returnBoriginal generationB	orchestraB	or schoolBor dieBops redBoperatorBoperative noBoperation tangoBoperation blackoutBoperation anchorageBopen teeBoogies revengeBoogiesBoogaBonrushBonline tamrielBonline summersetBonline hollowBonline greymoorBonline elsweyrBonline arcadeBonline alicizationBonline adventuresBoninakiBoniBonee chanbaraBoneeBonechanbara bikiniBone wayBone stepB	one punchBone againstB	on terrorBon marsB	on gorgonBon eridanosBon earthBomnoBomerta cityBomertaB	omensightBolympusBold mansBold ironBold godsBohsirBoffroad wideBoffice managerBoffice bustBof zodiarcsB	of xilliaBof wrestlemaniaBof worldB	of veniceBof troyBof starBof stalingradB
of spinnerB	of spartaBof skyBof separationBof sammunmakBof sailBof runersiaB
of refrainBof redB
of rebirthBof rallyBof rainBof radianceB	of rachelBof promathiaBof princessBof penBof peaceBof pbBof parisBof paradiseBof painB
of outlawsB
of olympusB
of norrathBof monstersB	of mirrimBof middleearthBof metalB	of memoryBof meatballsBof meBof mattBof lostBof loreBof loathingBof lineB	of kunarkBof juneBof jimmyBof jackB
of islandsBof isisBof influenceB	of infamyB	of icicleB	of icarusB	of horrorB
of hazzardBof hanBof grimrockBof gokuBof godsB
of gahooleBof forliB	of flightB
of firetopB	of familyBof exileBof edenB
of dunwallBof doomBof destiniesB
of despairB	of defeatBof deathBof daysBof danaBof conanB
of commandB	of colorsBof cardsBof britanniaB
of britainBof bootyBof boomstickBof berseriaBof bearsworthBof atlantisBof ariandelB
of ardaniaB
of arcadiaBof apokolipsBof alchemistBof akuBodyssey theBodin sphereBodinBode toB
oddysee hdBoddworld soulstormBoctopath travelerBoctopathBoctodad dadliestBoctodadB
octahedronBocean firstBobservationBobliteracersBobbBoath inBnutty adventureBnuttyBnumbersBnukem manhattanB	nukaworldBnuclear throneBnstrikeBnppl championshipBnpplBnoxB
nowhere toBnowhere prophetBnovelBnothing everBnot tonightB	not takenB
not enoughBnosurgeB	northgardBnorrathBnormal lostBnormalBnoneBnom nomB
nom galaxyBnobyBnobody knowsBnobody explodesBno yaibaBno rondeBno gameBninjago movieBninja xBninja saviorsBninja reflexBninja kinectB	ninja endBninja destinyBninja bladeBninetynine nightsB
ninetynineBnine parchmentsBnileBnights intoBnights enhancedBnightmare packBnightmare beforeBnightcasterBnight swordcraftBnight sorryBnight collectionBnight championB
night 2004B	nigh partBnicktoons uniteBniceB
nhl hockeyBnhl faceoffBnhl eastsideB
nhl arcadeBnhl 2k8Bnhl 2k5Bnhl 20Bnhl 19Bnhl 18Bnhl 17Bnhl 16Bnhl 15Bnhl 14Bnhl 13Bnhl 12Bnhl 11Bnhl 10Bnhl 09Bnfl tourBnfl quarterbackBnfl qbBnfl primetimeB
nfl arcadeBnfl 2k5Bnfl 2k3Bnfl 22Bnfl 21Bnfl 20Bnfl 19Bnfl 18Bnfl 17Bnfl 16Bnfl 15Bnfl 12Bnfl 11Bnext encounterB	next doorBnext chapterBnex machinaBnexBnewerthB	new tokyoB	new robotBnew moonBnew godBnew championsBneverending nightmaresBneverendingB	neverdeadB	never dieBneutron boyBneutronBnetwork racingB	network 6B	network 4Bnero nothingBneroBnerf nstrikeBnerfBneptunia rpgB
neptune vsBneptuneBneopets puzzleBneo theBneo cabBnelly cootalotBnellyB	nelke theBnelkeBneighborhoodBnedB
necropolisBnecromunda underhiveBnecrobaristaBnba unrivaledB	nba hoopzBnba basketballB
nba 2nightBnba 2k9Bnba 2k5Bnba 2k22Bnba 2k17Bnba 2k16Bnba 2k15Bnba 2k12Bnba 2k11Bnba 2kBnba 09Bnba 06Bnazi zombieBnaziB	naval opsBnatural doctrineB
nascar theBnascar dirtBnascar 2011Bnascar 2005Bnascar 2001B	nascar 09B	nascar 07B	nascar 06BnarwhalBnaruto uzumakiBnaruto pathBnarcosisBnarcB	napoleonsBnapoleon dynamiteB	nanostrayBnabooBn rollBn burnBmyth iiBmystic heroesBmystery journeyBmystery caseBmysterious trilogyBmysterious paintingsBmysterious journeyBmysteries ofB	mysteriesBmysims skyheroesBmysims racingBmysims partyBmysims kingdomBmy wordBmy rideBmy lordBmy lifeBmy katamariB	my friendB	mxgp2 theBmxgp2Bmxgp proB	mxgp 2019Bmx unleashedBmx 2002Bmvp 06BmuzzledB	mutazioneB
mutants inBmutantsBmutant stormBmutant meleeBmutant alienBmutant academyB	must fallBmushroom menBmusashiB	murder byBmuramasaBmundaunB
mummy tombBmummy demasteredB
multiverseBmuggedB	mudrunnerBmud fimBmudB	ms pacmanB
mr torguesBmr takoB	mr shiftyBmoxxisBmovie videoBmove editionB
mousecraftB	mountainsBmotorstorm arcticBmotorsport 7B	motogp 21B	motogp 20Bmotogp 2B	motogp 19B	motogp 13Bmotogp 0910B	motogp 07Bmotocross worldBmotocross maniaBmotionsportsBmothership zetaB
mothershipBmosaicBmorty virtualBmortyBmoriBmordheim cityBmordheimBmoralesBmoon stealthBmoon magicalB
moon diverBmoon 2Bmonstrum noxBmonstrumBmonstrous adventureB	monstrousBmonsters ultimateBmonsters meleeBmonsters jokerBmonster worldBmonster labB
monochromaBmonksBmonaco whatsBmonacoB
mogul 2007BmocoBmobB	mlb frontBmlb 14Bmlb 13Bmlb 12Bmlb 11Bmlb 10Bmlb 06BmisterBmisadventures ofBmirrimBmirage sessionsB	minute ofBminionsB
mini metroBminecraft xboxBmine theBmindjackBmindgate conspiracyBmindgateBmillionaires conspiracyBmillionairesBmilitaryBmiles moralesBmileBmilanoirBmiitopiaBmighty gunvoltBmighty gooseBmidtown madnessBmidtownBmicrobotB
mickey andBmiami takedownBmiami 2BmetronomiconB
metro 2033Bmetal headonBmetal blackB
messiah ofBmessiahBmeruruB
mercury hgB	merchantsBmerchantBmercenaries sagaBmenaceBmen rtsBmen inBmemento moriBmelodies ofBmelodiesBmegaton editionBmegagunB	megaforceBmegadimension neptuniaBmegadimensionBmedievilBmedieval warfareBmedieval totalBmedieval iiB	meatballsBme whyBme mrBme andBmcilroy pgaBmcilroyBmcgraths offroadBmcgrathsBmazeBmayanBmaximum impactBmaximum destructionBmaximoBmaxdB
max damageBmax 2B	matter ofBmatryoshka withB
matryoshkaBmasters seriesBmaster rebootBmaster chiefBmassive chaliceBmassive assaultB	maskmakerBmary skelterBmaryBmarvel tradingBmarvel avengersB	marrakeshB	mario theBmario strikersBmario sportsBmario rabbidsBmario powerBmario galaxyB
march partBmarathonBmaraBmaquetteBmanyBmanual samuelBmanualBmans journeyB	mans handBmans 24Bmanifold gardenBmanifoldB
mania plusB
manhuntersBmanhattan projectBmandatory happinessBmandateBmanager 2021Bmanager 2019Bmanager 2017Bmanager 2008Bmanager 2005Bmana remasteredBman zxBman confessionsBman aBmamodo battlesBmama 2B
mall brawlB	maliciousBmaldita castillaBmalditaBmaking historyBmakingBmaker 2Bmajoras maskBmajorasB	majin andBmajinB	majesty 2BmaizeBmain buildingBmainB	magneticaBmagnetic cageBmagneticBmagna cartaB	magicka 2Bmagical worldBmagical questBmagical melodyB
magic zeroBmagic circleB
mages taleBmage knightBmadness battleBmadness 2005Bmadness 2004B
madness 08B
madness 07B
madness 06Bmade manBmadeB
mad ridersBmachina deathBmace griffinBmaceBlydie suelleBlydieBlycorisBluxorBlustBlunar silverBluminous arcB	lulua theBluluaBluigi superstarBluigi bowsersBlufiaBludlums theBludlumsBluciusBlucidityBlucanorBlotusB	lostwindsB
lost worldBlost seaB
lost quarkB
lost phoneBlost kingdomsBlost judgmentBlost frontierBlost dimensionB
lost childBlost chaptersBlost andBloreBloot rascalsBlooseBlookoutBlongingB
long reachBlong myBlonesome roadBlonesomeB
lone sailsBlollipop chainsawBlollipopBlogyBlode runnerBlodeB
locoroco 2BlockBloathingBloadoutBlivingB	live rockB	live 2002Blive 19Blive 18Blive 16Blive 15Blive 14Blive 10B
little redBlittle onesBlittle dragonsBlittle 2B
lions songBlink evolutionB
line riderBline ofBlilo stitchBliloBlilies quietusBliliesB	like wineBlightyear ofB
lights outBlights cameraB
light fallBlife twoBlife theB	life goesBlife asBlife 2BlieBlibraryBliberty cityB	liars andBliarsBliar princessBletterBlethal skiesBlethal allianceBlegrand legacyBlegrandBlego ninjagoBlego marvelsBlego dimensionsBlego buildersBlego battlesB
legions ofB
legions dxBlegends definitiveB	legends 2Blegendary alchemistsB
legacy theBlegacy taleBleft behindBlederer allBleague soccerB	league iiBleadsBleaderBlead theBlead andBlaytons mysteryBlaytonsBlawnBlawbreakersB	launch ofBlaunchBlast tinkerBlast rewardBlast resortBlast recodeB	last doorBlast dayBlast campfireB
last bladeBlaser leagueB	larry boxBlapis xBlapisBlantern riseB
langrisserBlands ofBlandmark editionBlandmarkB
land touchBlancerBladyBlacrimosa ofB	lacrimosaBla rushBla killBla copsBkuzunoha vsBkuzunohaBkunarkBkunaiBkorpsB	kororinpaB	korg ds10BkorgBkoreaBkongaBkong tippingBkong racingB
kong minisBkong massacreBkong jungleBkonaB	kombat vsBkombat shaolinBkombat arcadeB
kollectionBkobuto vBkobutoBknowsBknowledge isB	knowledgeBknights questB
knights inBknights contractBknight swordBknight sagaBknight kingBknight chroniclesBknight aBknife ofB	knee deepBkneeBknackBklonoa 2BkiwiBkiwami 2Bkiss psychoBkirby fightersBkintaros revengeBkintarosB
kings birdB
kings 2002B
kings 2001Bkingdoms xivBkingdoms xiB
kingdom ofBkingdom battleB	king riseBking legacyBking aB
kimetsu noBkimetsuBkim possibleBkimBkillzone shadowBkilling harmonyBkiller7Bkiller instinctB
killer appB	kill teamBkill laBkill ifB
kid icarusBkick heroesBkeyBkengoBkena bridgeBkenaBkeflingsBkeepersBkeep talkingBkeaneBkatrielle andB	katrielleBkatana zeroBkatBkasumi stolenBkasumiBkartingBkarakuriBkanes wrathBkanesBkaitosB	kain soulBkagura shinoviBkagura peachBkagura estivalBkagura burstB
kagura bonBkabutoBjydgeBjustice aceBjunkiesBjungle beatBjuneBjoy rideBjourney katrielleBjourney downBjoint strikeBjoint operationsBjoe redBjoe operationB
joe deversBjoe 2BjockeyBjimmy neutronBjimmyBjett theBjett rocketB	jet grindBjeremy mcgrathsBjenga worldBjengaBjedi starfighterB
jedi powerBjay andBjayBjanesBjam vendettaBjam sessionsBjam rapstarBjam onBjam maximumBjam iconBjakeBjade empireBjack sparrowB
jack keaneBizunaBix monstrumBivy theBivyBiv shiveringB
iv knightsBiv conquestBiv bloodB	ittle dewBittleBit outBit movesBislands toweringBisland thunderBisland specialBisisBisaac rebirthBisaac afterbirthBis youBis powerBis noBis mineB	is enoughBirony curtainBironyB	iron kingBiron iiiBiron iiBiron brigadeBiridionBioBinvizimals theB
invisiblesBinvisible warBinvisible incBinvincible tigerB	inversionBinvaders infinityBinvaders foreverBinto dreamsBinternational winterBinternational tennisB
intergradeBintentBintellivision livesBintellivisionBintelligence quotientB
insurgencyBinside storyBinsecticideBinmostBink machineB	influenceBinfliction extendedB
inflictionBinfinity geneBinfinity 30Binfinity 20Binfinite minigolfBinfestationBinferno poolBinertiaBindustryBindianapolis 500BindianapolisB
inc screamBinc megaBinc aBinbirth exelateclrBin spaceB	in sheepsBin otherBin nightmaresB	in motionB
in monsterBin maraBin manhattanBin felghanaBin disguiseBin conflictBin blackBimmortal unchainedBimmersion racingB	immersionBillusion starringBikenfellB	iii thirdBiii riseBiii eternalB	iii ashesBii wingsBii totalBii operationBii heartBii hdBii fleshBii egoBii deadfireBii darkB
ii crimsonBii childrenBii 2005BigiBif foundBiconBicoBicicle creekBicicleBiceyBibb obbBibbBi itsBi expectBi doBhysteria hospitalBhysteriaBhyperparasiteBhyper scapeBhydrophobia prophecyBhydorahBhybridBhyakki castleBhyakkiBhustle kingsBhunting simulatorBhunting groundsBhunters legacyBhunters daybreakBhunter nowhereBhunter generationsBhunter allianceBhunter 3Bhunter 2Bhunt heartsB
hunt bloodBhundred yearsBhumans 2Bhulk deathwingBhousesBhour ofBhottest partyBhottestBhotel giantB
hot rumbleBhostileBhospital emergencyBhoshigami ruiningB	hoshigamiBhorus stationBhorusBhorizonsBhorizon legacyBhordesBhoraceBhoopzB	hoops 2k8B	hoops 2k5Bhonor vanguardBhonest heartsBhonestBhomestar ruinerBhomestarB	homecourtB	holmes vsBhollow knightBhollow fragmentBholeBhockey leagueBhoardBhoaBhiveB	hitman goBhistoriaBhinokami chroniclesBhinokamiBhill originsBhill hdBhill downpourBhill 3BhikeBhideoutBhiBhgBhexicBherolandBheroines tagB	heroes viBheroes talesBheroes rebornB
heroes iiiB	heroes iiBheroes chroniclesBhero vanBhero unsignedB
hero superBhero nobodyB	hero liveBhero iiBheritage ofBheritageBheretic kingdomsBhereticBheraclesBher piratesB
helsing iiBhelmet chaosBhelmetB
helldiversBhellbound debriefedBheirBheinz wolffsBheinzBhedge hammyBheavyweight boxingBheavyweightBheavy weaponBheavy metalBheavenswardBheavenlyBheat 5Bheat 4Bheat 3B	heat 2002B
heartslashBhearts melodyBhearts flamesB
heartbreakBheart leadsBheart 3BheadonB
headlanderBheadhunter redemptionBhd verB
hd trilogyBhd remasteredBhazzard returnBhazzardBhazeBhazard bloodBhave feelingsBhaveBhassleBhasbeen heroesBhasbeenB
hardy boysBhardyBhardcoreBhardboiled chickenB
hardboiledB	hard rockB
hard corpsBhappy havocBhappiness freakpocalypseBhandsome collectionBhan taoBhanBhamtaroB
hammy goesBhammyBhammerlocks bigBhammerlocksBhammerBhamiltons greatB	hamiltonsBhamBhalo theB
halo reachBhalfminute heroB
halfminuteB	half pastBhalfBhalenBhakushoBhackgu lastBhackersBhackBh1z1B
gyromancerBgx tagBguzzlers extremeBguys ultimateBguy gameBguy backBguruminBgunstarBgunship eliteBgunshipBgunpeyB
gungriffonBgundam extremeBgundam 3B
gun combatBguideBguardians theBguardian heroesBguardBgti clubBgtiB	grow homeBgromit curseB	gripshiftBgrind radioBgrimrockBgriffin bountyBgriffinBgrid theBgreymoorBgreen lanternB
green hellBgreatestBgreat adventureBgrayBgravity ghostBgravity crashB
gratuitousBgrand kingdomBgrand battleBgraceful explosionBgracefulBgothic armadaBgothic 4Bgothic 3BgorogoaBgorgonBgoodbyeBgone toB	gone sourB	gone homeB	golf withB	golf openBgolf 2BgolemB
gold gangsBgokuBgoes theBgoes onB	goes nutsBgodzilla saveBgodzilla destroyBgods redemptionBgodfallBgoddess innerBgod warsBgod modeBgo vacationBgo homeBgo definitiveB	go chronoBglory ofBglobal strikeB
gladiatorsBgiveBgitaroo manBgitarooB
girl fightBgiraffe andBgigsBgig riseBgigBgiants citizenBghost riderBghost inBghost bladeBgforceB
gettysburgBgetawayBgeoBgenso wandererBgenshin impactBgenshinBgenjiBgenius attackBgenieBgenerator 3Bgeneration ofBgeneralsBgeneral leeBgemini heroesBgemini 2BgekidoBgears tacticsBgearclub unlimitedBgearclubBgear x2Bgear xBgear onlineB	gear acidBgazeBgauntlet sevenBgato robotoBgatoBgatling gearsBgatlingBgathering stormBgathering battlegroundsBgates ofBgate andBgaryBgarfieldBgardening mamaB	gardeningBgarageBgames presentsBgames ofBgameday 2002Bgameday 2001B
game wrongB	game nerdB	game huntBgambitBgallowsBgalleryBgalgun 2B	galaxy onB
galakz theBgalactica deadlockBgalactic battlegroundsBgahooleBgaea missionBgaeaBgabriel knightBgabrielBg racingBfuzion frenzyBfuzionBfuturidium epB
futuridiumBfuture toneB	future 88BfuturamaBfutbolBfusion raceBfuseBfury battleBfuruBfurious crossroadsBfur fightersBfullyBfull hdB
full burstB	full bodyBfuga melodiesBfugaBfuel ofBfruit racingBfruit ninjaBfrostpunk consoleBfrontsBfrontlines fuelB
frontlinesBfrontier remasteredBfront officeB
from spaceBfrom monkeyBfrom matryoshkaB	from edenBfrom butcherBfrogger helmetB	frogger 2BfritzB	fright ofBfrightBfriend pedroB	fret niceBfretBfrenzy 2BfreezeBfreedom forceBfreedom fingerBfreedom cryBfreakyformsBfreakpocalypseBfrance 2013Bframe iiBfragmentBfractured maskBfractureBfox 64B
fowl fleetBfowlBfourths homeBfourthsBfour swordsB	four 2001B
foundationBfosterB
fortune iiBfortune andBfortixBforsaken kingdomBformula oneBformBforliBforever afterB
foreclosedBforces teamBforcedBforce xtremeBforce worldBforce editionBforce bowlingBforce anniversaryB
force 2025B
force 2017B	forbiddenBforagerBfor suburbiaB	for nabooB
for kinectBfor gravityB
for europeB	for earthBfor atlantisB
for asgardB
for answerBfootball 2k8Bfootball 14Bfootball 13Bfootball 12Bfootball 11Bfootball 10Bfootball 09Bfootball 08Bfootball 06Bfootage volBfootageBfoodBflying fortressBfluidityBflipping deathBflippingBflipperBflipB	flinthookBflesh bloodBfleshBflatout ultimateB	flatout 4B	flames ofBflagsBflag freedomBfitness systemBfitness boxingBfists ofBfistsBfishing masterBfishermans taleBfirst toBfirst templarBfirst departureBfirst contactBfiretop mountainBfiretopBfirefighterBfire warriorBfire iiBfire editionBfinal stationBfinal showdownBfinal prototypeB
final examBfinal battleBfinal 2BfimbulBfim motocrossBfimBfilter darkBfilmBfiles tunguskaB	files theBfigmentBfightsBfighting climaxBfighters xiiiBfighters xiiBfighters maximumBfighters 0203Bfighter anniversaryB	fighter 4Bfight streetwiseBfight doubleB
fight crabB
fight clubBfifa 22BfieldsBfield ofBfestaBfesBferrari racingBfenixB	fell sealBfellBfelghanaBfeistBfeelings tooBfeelingsBfeeding frenzyBfeedingBfateextella theBfateextella linkB
fateboundsBfate hdBfatalisBfatal inertiaBfast racingBfast andBfarewellB	far shoreBfar loneBfantasy zoneBfantasy xiii2Bfantasy theBfantasy iiiB
fantasy iiBfantasy heroBfantasy fablesBfancy pantsBfancyBfamily bloodBfamicom detectiveBfamicomBfalseBfallout brotherhoodBfallen enchantressB	fall guysB	falcon 40Bfairytale fightsBfairBfaeriaBfade toBfadeB	factory 4BfactionsBfaction battlegroundsBfaceoffBfaceBfable fortuneBf355 challengeBf355Bf1 2009Bf adventBf 2ndBezio collectionBezioBeyetoy playB	extreme gBextreme exorcismB	extreme 2Bextend extraBexplosion machineBexplorer warriorsBexplodesBexplainBexpendables 2BexpendablesBexpeditionsB
expect youBexpectB	expatriotBexorcismBexodus completeBexit theBexist archiveBexistBexiledB
exelateclrBexceedBexamBex theBex invisibleB	ex cursedBevolandBevil outbreakBevil geniusBevil editionBevil archivesBeverybodys goneBeverybodys golfBevery extendBevery buddyBeverquest onlineBeverhoodBever remainsBever darknessBeve valkyrieB
eve onlineBeve ofBevasionBevans remainsBevansBeufloriaB
etherlordsB	ether oneBetherBeternity iiBeternal sonataBeternal nightBeternal castleBestival versusBestivalBesther landmarkB
espn majorB
escha logyBeschaBescapevektorBescape virtuesBeridanosBepisodesBepisode ultraB	epic yarnB	ep deluxeBeoBenhanced editionsBengagedBenforcerBenemy engagedBendless spaceBendless oceanB	enders hdBender liliesBenderBend requestB
encountersBenclaveBenchiridionBenchantressBenchanted armsB	enchantedBempire dividedB	empire atBempathyBemergency wardBemblem warriorsBemblem threeBemblem shadowB	embers ofBembersBelysian tailBelysianBelsweyrB
elite wwiiBelite vrBelite squadronB
elite naziBeleven 7Beleven 2B
electronicBelebitsBeldritchBeldest soulsBeldestBeightBegyptianBego draconisBeffect overdoseBeffect connectedBeetsBeditionsBedition remixBecw anarchyBecwBecoBecks vsBecksBeater 2Beat leadBeastwardBeastside hockeyBeastsideBeastB
earthnightBearthlock festivalB	earthfallBearth iiBearth atlantisBearth assaultBeagle flightBea playgroundBdynasty tacticsBdynamite theBdynamiteBduty classicBdust anBdusk skyBdusk seaBdungeons ofB
dungeons 2Bdungeon hunterBdungeon explorerBdungeon everyBdungeon brosBduneBdukes ofBdukesBdugBduelist linkBduelistBds10BdrugBdroplitzBdrome racersBdromeBdrive offroadBdrive ferrariB	drive eveBdriv3rBdriller drillBdrew dangerBdreamworks madagascarBdreamscaperBdreamfall theB
dream landBdream chroniclesBdrakesBdragons darkBdragons daggerdaleBdragons cafeB	dragonflyBdragon starBdragon neonBdragon knightB	dragon ivBdragon emperorBdragon bladeBdracula theBdraconisBdr nedBdozenBdownward spiralBdownwardBdownpourBdownfallB	down teamBdown chapterBdouble kickBdouble helixBdoraemon storyBdoraemonBdoom vfrBdoom iiBdoom 64Bdonut countyBdonutBdont tryB	dont stopBdont dieBdonkey kongaBdonB	dominionsB	dominatorBdolphin defenderB	doll kungBdokapon kingdomBdogs definitiveBdodogoBdodgeball remixBdodgeball adventureBdodgeball academiaBdo toBdo notBdnaBdlc packBdlcBdkBdiyBdivision undergroundBdiva futureBdissidia finalB
disneys pkBdisneys liloBdisneys kimBdisneys atlantisBdisneys aladdinBdisneyland adventuresB
disneylandBdisney singBdisney magicalBdisney castleBdisjunctionB
dishwasherBdishonored definitiveB	disgaea 5B	disgaea 3B	disgaea 2B	disgaea 1Bdisco dodgeballB	disc roomBdisc jamBdisappearanceBdirt toBdinotopia theB	dinotopiaBdinosaur hunterBdimensionalBdillons rollingBdigital devilBdigimon rumbleBdig dugBdig 2Bdie hardB	die aloneBdie 3Bdie 2B
diddy kongBdiddyBdid iBdidBdice ofBdice legacyBdiaryBdiariesBdew 2BdewBdevils cartelB
devil sagaBdevers loneBdeversBdetroit becomeBdetective clubBdetective 2Bdestruction derbyBdestiny riseBdestinationBdesperate struggleBdesperate escapeBdespair girlsB
desolationBdeserve thisBdesert ratsB
descendersBderbyBdepthsB	departureBdengeki bunkoBdengekiBdenBdemons soulsB	demons ofBdemon slayerB
demon gazeB	democracyB	dementiumB
demasteredB
degrees ofB	deformersBdefinitive seriesBdefaultB
deep blackBdecay 2BdecadesB	debriefedB	deathwingB	deathtrapBdeathsmilesBdeaths gambitB	deathloopBdeath trackB
death roadBdeath machineB	death endBdeath byBdeadly towerBdeadly shadowsBdeadly dozenBdeadlockBdeadlight directorsBdeadliest warriorBdeadfireBdeadfall adventuresBdeadfallBdeadcoreBdead synchronicityBdead saintsBdead revolverBdead regenerationBdead onslaughtB
dead kingsBdead iiiBdead effectB
dead blockBde amigoBdays ofB	days goneBdaybreak specialBdaybreakBdaxter collectionB	dawnguardBdateBdataBdash sizzleBdaryl deluxeBdarylBdarwiniaBdarwinBdarq completeB	darkwatchBdarkstar oneBdarkstalkers resurrectionBdarkestville castleBdarkestvilleBdarkened skyeBdarkenedBdark tomorrowBdark resurrectionBdark mirrorBdark messiahBdark matterB
dark forceBdark dreamsBdark devotionBdark cornersB
dark cloudBdariusburst chronicleBdangerous drivingBdangerous 2Bdangeresque 3BdangeresqueBdanganronpa v3Bdanganronpa triggerBdanganronpa anotherBdandyBdandaraB
dancing inBdancerB
dance beatBdance 3B
dance 2019B
dance 2018B
dance 2015B
dance 2014BdanaBdale enhancedB
daggerdaleBdaggerBdaemon xBdaemonBdadliest catchBdadliestBdacBd4 darkBd4Bd2BcycleBcybeastBcyanide happinessBcyanideBcut itBcustom roboBcustomBcurtain fromBcursed kingBcursed castillaB	curse theB
curse partBcurious georgeBcurious expeditionB
cup brazilBcup 2011Bcultist simulatorBcultistBcthulhu tacticsBcthulhu darkB
ct specialBctBcrystal menhirBcrystal defendersBcry vergilsBcry definitiveB
crush hourB	cruise spBcrown trickBcrossing newB	crossfireB	cross tagBcrooked mileBcrookedB
croft tombBcriticalBcrisis 3Bcrimson skiesBcrimson seaBcrimson butterflyBcriminal projectiveBcriminal originsBcriminal girlsBcrimewave editionB	crimewaveB
crimecraftBcreed championsBcreations aliveB	creationsBcreaksBcrash twinsanityBcrash nB
crash dashBcradleBcrackdown 3BcrabBcovenantBcoven ofBcovenBcoveBcountyBcountry tropicalBcountry returnsBcount lucanorBcountBcotton rebootBcottonBcossacks iiBcorps uprisingB
corners ofBcornersBcorner shopBcore verdictBcore vB	core plusBcore forBcore 4Bcore 2Bcootalot theBcootalotBcooper thievesBcool boardersBcookie creamBcookieBcookB
convictionBcontrol ultimateBcontrol theBcontrol aweBcontrastBcontra anniversaryBcontemplationBconstantineBconquer theBconquer generalsBconnection cornerB	connectedBconnectBconfrontationBconflicts vietnamBconflicts secretBconflict zoneBconfidentialBconfessions ofBconfessionsBconduitBcondition coloniesBcondemned criminalBcondemned 2Bconcrete jungleBconception iiBcommits taxBcommitsB
commando 3B
commandersBcommander 2Bcommand missionBcommand gaeaB	command 2Bcomes todayBcombat zonesBcombat leagueBcombat flightBcombat firstBcombat eliteBcombat commandBcomancheBcoma 2B	colosseumBcolorful taleBcolonies editionBcoloniesBcollection plusBcollection ofBcollection consoleBcollection arcadeBcollection 1B	cold soulBcoffin dodgersBcoderealize guardianBcoderealizeB
coded armsBcodedBcode ofBcocotoBcobaltBcoach 09BcoBclubhouse gamesB	clubhouseBclub theB	club plusBclub losB	club 2002B	club 2001Bclub 2Bcloudy withBcloudyBcloudsBclothingBclosureBclosedBclose quartersBclone inBclimberBcliffsBclid theBclidB	cleopatraBclearBclawsBclassics anniversaryBclassic trilogyBclassic arcadeBclassBclaptraps newB	claptrapsB
clank sizeBclank collectionBcivilization beyondB
city girlsBcitizen kabutoB	cities ofB	cities inBcircularB
chronologyBchrono stonesBchronicles oneBchronicles myBchronicles iiBchronicles echoesBchronicles 3Bchronicle savioursBchroma squadBchromaBchristmas oogiesBchristie andB
chosen oneBchopBchocobos mysteryBchivalry medievalBchinatown warsB	chinatownBchimpBchief collectionBchiefB	chicory aBchicoryBchicken runBchickBchef brigadeBchefBcheerB	checkmateBcheatsBcheat emBcheatB	chase forBchargedBchapters bookBchapter oneB	chapter 4BchaoticBchaos risingB
chaos riotBchannel battleB	chance ofBchanceBchanbara originBchanbaraBchampionship tournamentBchampionship throwdownBchampionship surferBchampionship paintballBchampionship 2011BchaliceBchainsawBchain ofBcenturyBcentral fictionBcenter underB
cent bloodBcellfactor psychokineticB
cellfactorBcell convictionBcelebrationBcdBcayBcatherine fullBcaterpillarsBcatanBcat andBcastlevania theBcastlevania symphonyBcastlevania curseBcastlevania advanceBcastle remasteredBcastilla exBcastBcartaBcartBcars raceoramaBcarrier commandBcarrierBcarolBcarmageddon maxBcargoBcareerBcaptain tsubasaBcaptain scarlettBcapsizedBcapcom originsBcannonB
cane magicBcaneBcandleBcanadaBcan eatBcampfireBcampaign ofBcampBcamerons darkBcameron filesBcameronBcameBcallingB	cake bashBcage closedBcaesarBcabB
by numbersBby flameBbuzz theBbutton cityBbuttonB	butterflyBbutcher bayBbusinessBbushidoBbush hockeyBbus simulatorBbury meBburyBburtons theBburtonsBburst limitBburst battleBburnout legendsBburnout dominatorBburnout crashB	burnout 3Bburning bridgesBburning bloodBburner climaxB
burgertimeBbunko fightingBbunkoB
bumblebeesBbullet witchBbullBbuildings haveB	buildingsBbuildingBbuilders journeyBbugsB
budokai hdB	budokai 2B	bubsy theBbruisedBbrotherhood theBbros uBbros forBbrokerBbroken worldBbroken toysBbroforceBbritneys danceBbritneysB	britanniaBbritainBbrigmore witchesBbrigmoreBbrigandine theB
brigandineBbridgesB	bridge ofBbridge crewBbride ofBbreed impactB
breathedgeBbreakout 10thBbrazilBbrawloutBbravely defaultB	brave newB	brave andBbrave aBbrain challengeBbrain academyB
boys miamiBboyfriend dungeonB
boy geniusBboy foreverBboy commitsBboy ashaBboxing roundBboxing creedBboxersBbowsers insideBbourne conspiracyBbourneBbound byBboruto expansionBborderlands gameBborderlands claptrapsB	boomstickBboomerang xB	boomerangB	boom bloxBbookworm adventuresBbookwormBbooksBbookbound brigadeB	bookboundBbomberman maxBbomberman liveBbold theBboktaiB	bodycountBbob mallBboarders 2001BboardersBblue vsB	blue fireB
blue earthBbloxB	bloodshotBbloodrayne betrayalB	bloodlineB
bloodborneB
blood tiesBblood onB
blood bathB	blood andBblocksBblizzard arcadeBblinxBblind princeBblight belowBblightBbleeding edgeBbleedingB
bleach theBblazblue crossBblazblue centralBblast 2Bbladestorm theBbladed furyBbladedBblade iiBblade hdB	blackwellBblacksite areaB	blacksiteBblacklight retributionBblackguardsBblack versionBblack legendB	black katBblack futureBblack bruisedBbittrip fateB	birthdaysBbirth ofBbipedBbiohazard bannedB	billiardsBbikeBbigs 2Bbigger boulderBbiggerBbig televisionsB
big rumbleBbig conB	big brainBbig adventureBbfeBbeyond gothamBbeyond earthBbetterBbetrayerBbethesda pinballBbethesdaBbestBberseriaB	bendy andBbendyBben 10BbenBbelievinBbejeweled twistBbejeweled 2Bbeholder completeB
behind theBbefore christmasBbedlamBbecome humanBbeautiful desolationBbeatsB
beat saberBbeat hazardBbeat copBbeast riderBbearsworth manorB
bearsworthB
bear panicBbboyBbayonetta vanquishBbayBbattletoadsBbattlestations pacificBbattlestations midwayB
battles ofBbattlelinesBbattlefront eliteBbattlefleet gothicBbattlefleetBbattlefield commandoBbattlefield 2142Bbattlefield 1943Bbattleblock theaterBbattleblockBbattle realmsBbattle marchBbattle islandsBbattle fantasiaBbattle chefBbattle brothersB	battle atBbattle assaultBbattle archivesBbattalion warsBbatman darkBbatman 3Bbath andBbathBbaten kaitosBbatenBbasketball 10Bbasketball 09Bbaseball riotBbaseball 2k9Bbaseball 2k7Bbaseball 2k5Bbaseball 2k3Bbaseball 2k12Bbaseball 2k11Bbaseball 2k10B
baseball 2Bbasara samuraiBbasaraBbaronB	banner ofBbanned footageBbannedB
banjotooieBbang gunshipB
band trackB
band blitzBband 4Bballs ofBballsBballers phenomBballers chosenBballadBball originsBball deluxeBbakugan defendersBbadlandsBbadland gameBbadlandB	badia theBbadiaB
baddest ofBbaddestB	bad northBbad mojoBbad boysBbabel risingBbaba isBb17Bazure 2Bazkend 2BazkendBayeshaBaxis ofB	axe beastBawesomenauts assembleBaweBaviary attorneyBaviaryBavengers battleBavengerBauto libertyBauto iiiBauto chinatownB	audiosurfBatv supercrossB
atv reflexB	atv aliveBattorney trialsBattack expatriotBattack 2Batomic tankBatomic ninjasBatomBatlas muggedBatlantis theBathletics theB	athleticsBateliers ofBateliersBatelier shallieBatelier mysteriousBatelier meruruBatelier lydieBatelier luluaBatelier eschaBatelier ayeshaBatari classicsBatari anthologyBat homeBasuras wrathBasurasBastria ascendingBastriaBassembleBassault squadBassault heroesBassassins ofBashenBasha inBashaBash ofBascentB	ascendingBas aBarx fatalisBarxBartifactB
arthur andBartful escapeBartfulBarmored killB	armed andBarmedBarma iiBarkanoidBariandelBariaBarena ultimaxBarena 2BardaniaBarctic edgeBarchives volumeBarchives residentBarchive theBarchiveB	archangelBarcania gothicBarcaneBarcadiaBarcade stadiumBarcade kollectionBarcade collectionBarbiters markBarbitersBaragorns questBaragornsBaragami shadowB	aragami 2B
ar nosurgeBaquaman battleBaquamanBappBapotheonBapollo justiceBapolloB	apokolipsBapocalypse neverBapocalypse heartBapocalypse editionBape outBapbB	apartmentBao internationalBanother episodeB	anomaly 2Banniversary worldBannikaB
anima gateBangry videoBangel ofBand yetBand wineBand tribulationsBand thenB
and silentB
and nobodyB	and mortyBand merchantsBand herBand goldBand dangerousBand coB
and cheatsBand baldursB
and annikaBand 2Band 1BancientsBancient worldBancient warBancient epicBancient artsB	anchorageBancestors legacyBanarcuteBanarchy rulzBanarchy reignsB	an unbornBan interactiveB
an elysianB
an ancientBamyBamusement parkB	amusementB	amulet ofBamuletB	amplitudeBamong thievesBamnesia rebirthBamnesia collectionBamigoBamerican sk8landBamerican nightmareBamerican conquestB	ambitionsBambition sphereBam deadBalwasBalteredB
alter echoB
alone withBalone complexBalmost goneBalmostBallstars battleBallstar fruitBallpro footballBallproBalloyBalliance rageBalliance iiBalliance aliveBall zombiesBall youBall thatBall monstersBall inBalien syndromeB
alien rageBalien assaultBalicization lycorisBalicizationBalice inB	alexanderBalert 2Balekhines gunB	alekhinesBalchemists ateliersBalchemists andBalchemist andB
alan wakesBakuBakkadianBairforce deltaBairforceBairborne assaultBairaceBair warB
air attackBails youBailsBai theBages medievalBages 2Bagent clankB
age scratsBage actBagainst allBagainstBagain noB
afterbirthB
after fallBafrika korpsBaeternobladeBaerialBaereaB	aeon fluxBaeonBaegis defendersBadventures theBadventures 2Badventure ofBadventure islandBadventure hdBadventure gameBadvent risingBadvent darkBadvance theBadvance collectionBadr1ftBadams ventureBadamsBactraiser renaissanceB	actraiserBactivision anthologyB
across theBacidBachtung cthulhuBachtungB	acdc liveBacdcBaccelerationB
aca neogeoBacaBabsoluteB
absence ofBabsenceB	above theBabbaBa wonderfulB
a warriorsBa totalBa storyBa spaceBa shortBa puzzleB	a promiseBa pixelBa normalBa monstrousBa matterB	a machineBa linkB	a knightsBa fishermansBa darkBa cyberpunkB	a crookedB
a colorfulBa cloneBa climbBa chanceBa bigB96B8thB8bit isB	88 heroesB80sB8 xtremeB8 dlcB8 deluxeB76 wastelandersB	7 wondersB7 internationalB7 daysB	6 empiresB	6 cybeastB5thB500 legendsB500B5 wolvesB5 teamB5 riseB5 fromB5 finalB	5 empiresB5 dontB5 desperateB5 checkmateB5 8bitB	4x4 worldB4x4 mastersB42B
40000 killB
40000 fireB4000B40B4 whatB4 valleyB4 totalB	4 thickerB4 takeB4 summerB4 nukaworldB4 metamorphosisB4 mayhemB4 inB4 hdB4 goldenB4 faithB	4 empiresB	4 editionB4 dangeresqueB	4 burningB4 5B3d worldB3d theB
3d megatonB3d aB3d 20thB360 editionB360B
30 editionB3 whatB
3 tiberiumB3 thisB
3 takedownB3 resurrectionB3 pointB	3 piratesB3 onceB3 nhlB	3 muzzledB3 mothershipB3 marrakeshB3 leviathanB3 kanesB
3 judgmentB3 hdB3 fullB3 fiaB3 fateB3 fallenB3 deluxeB3 closeB	3 citadelB3 catchB3 blackB3 bfeB	3 baddestB	3 armoredB	3 absenceB3 aboveB2nightB
2nd runnerB2k22B2k playgroundsB28B25thB256B24 hoursB2142B20xxB2033B2025B2016 olympicB	2014 fifaB
2013 100thB2012 theB2012 olympicB2011 theB
2005 chaseB2002 featuringB
2001 majorB	20 killerB
20 editionB2 zerkerB2 wrongB2 wingsB2 wiiB	2 warmindB2 vrB	2 viciousB2 underB2 thereB2 sufferB2 strongB2 stormB2 starB2 sirB	2 severedB2 seedsB	2 revengeB2 remasteredB2 rageB2 plusB2 mrB2 lairB2 kasumiB2 inB2 heartB2 fiaB2 fallB	2 eternalB2 eoB	2 empiresB2 desperateB2 crimewaveB2 contemplationB2 caseB	2 captainB2 brideB2 bloodshotB2 biggerB2 battlelinesB2 atlasB2 assassinsB2 apartmentB1995B
1942 jointB1941B187 rideB187B1701B14 theB13 theB11 theB11 aftermathB10th anniversaryB101in1B101 remasteredB1 streetballB1 launchB
1 homestarB1 frightB
1 completeB1 andB1 allB0910B06 totalB06 theB06 ncaaB03B0203B007 theB007 reloadedB007 legendsB	zx adventB	zur machtBzurBzuma deluxeBzumaBzuboBzotrixBzoonies escapeBzooniesBzool redimensionedBzoolB
zoo keeperB
zoo empireBzone massacreBzone iiBzombiuBzombies invasionBzombies chroniclesBzombie wranglersBzombie vikingsBzombie tycoonBzombie smasherBzombie slayersBzombie revengeBzombie panicBzombie ninjaBzombie nightBzombie hunterBzombie defenseBzombie burnB
zombie bbqBzombeerB
zoids wildBzoids assaultBzoeBzip lashBzipBzinBzill ollBzillBzigguratBzhp unlosingBzhpBzexal worldBzexalBzeus masterBzeusBzeta gundamBzestiriaBzerzuraBzero theB
zero reduxBzero noBzero missionB	zero lastBzero intrepidBzero iiiBzero iiBzero gravityBzero dBzero collectionBzero 4Bzero 3Bzeonic frontBzeonicBzeonBzenzizenzicBzenith 2016BzenithBzendokuB	zelda triBzelda spiritBzelda phantomBzelda linksBzelda iiBzelda collectorsBzeit squaredBzeitBzehirBzealand storyBzealandBzax theBzaxBzarvotBzarathustraB
zapper oneB
zanki zeroBzankiBzanB	zack wikiBz2 chaosBz2Bz tenkaichiB
z taiketsuBz puzzleB
z infiniteBz harukanaruBz forB	z extremeBz buusBz attackByuris revengeByurisByuppie psychoByuppieByupitergradB	yumis oddByumisByumenikki dreamB	yumenikkiByugioh zexalByugioh worldwideByugioh reshefByugioh nightmareByugioh dungeonByugioh capsuleByugioh 7B
yugioh 5dsBys viBys iBys booksByoutubers lifeB	youtubersByourselffitnessByouropaByoure inByoureB	your roleByour paradeByour ownB
your mouthB	your lifeB	your handB	your eyesByour enemiesByour castleB
your brainByour arsenalB
young thorByoungByoull returnByoullByou willB	you stealByou pikachuByou meB	you finalByou dropByosumin liveByosuminByoshis storyB
yoshis newByoshis craftedByoshi touchByoshi topsyturvyB
york timesByork invasionByorha editionByorhaByono andByonoByomawari theByomawari nightByomawari midnightByogByieldByie arByieByet anotherByestermorrowByellow brickByellow avengerByear theByear oneByear ofByear celebrationByawhgByatagarasu attackB
yatagarasuByars revengeByarsByanya caballistaByanyaByamadaByakuza deadByakuza 2ByahtzeeByager missionsByaga theByagaByBxyxxBxyanideB
xv windowsBxv royalB	xv pocketBxulimaB
xtype plusBxtypeBxtreme sportsBxtreme beachBxtreme 3Bxtend editionBxtendBxsquadBxrd signBxrd revelatorBxrd revBxpand rallyBxpandBxmorph defenseBxmorphB
xmen reignB
xl editionBxl 2012Bxl 2011BxixBxillia 2Bxiii centuryBxii revenantBxi treasuresB	xi echoesBxg blastBxgBxfiles resistBxfiles gameB	xfightersBxfBxeviousBxenoraidB	xenonautsB	xenogearsBxeditionBxd galeBxcom enforcerBxcom chimeraB
xbox musicBxanadu nextB	xanadu exBxanaBx7Bx6Bx5Bx4 foundationsBx4B	x3 terranB
x3 reunionBx2oBx2 theB	x2 reloadBx superbikeBx spaceBx skiesB	x saurianB	x rebirthBx ninjaBx nightmareBx megaBx masterBx maliceBx gamesBx combatBx chroniclesBx beyondB	x advanceBx accelerationB	wwii tankBwwii pacificB	wwii acesB	wwf royalBwwf roadBwwf rawBwwf noBwwe survivorBwwe roadBwwe rawBwwe 2k15BwuppoB
wunderlingBwulfBwtf workBwtfB	wtcc gameBwtccBwritheBwriterBwrestling worldBwrestling returnsBwrestling featuringBwrestlemania xixBwrestlemania 21Bwrecked revengeBwreckedB
wreckateerBwrc powerslideB	wranglersB
wraith theBwraithBwounded dragonsBwoundedBwoundBworms ultimateBworms rumbleBworms reloadedB
worms clanBworms aB	worm jazzBwormBworldwide editionBworldwide classicsB
worldshiftBworlds scariestBworlds onlineB	worlds ofBworlds kronosBworlds enterBworld vsBworld tacticalBworld superBworld stageBworld recordsBworld radiantBworld questB
world parkBworld myBworld missionBworld legendsBworld kitchenBworld invitationalBworld heroesBworld granadoB	world endBworld editionBworld eatersB
world duskB
world duelBworld dsBworld driverBworld dominationBworld destructionBworld defendersB
world dawnB
world dataBworld conquestBworld classBworld bowsersBworld betrayedBworld apartBworld amusementBworld adventuresBworld 4B
world 20thBworks buildBworksBworkoutsBworkoutB	work timeBwordjongBword puzzleB
woolfe theBwoolfeBwookieesBwoody woodpeckerBwoodyBwoodpecker racingB
woodpeckerBwooden senseyBwoodenB
wood elvesBwonders shadowB
wonders iiBwonders beyondBwonderful journeyBwonderbook diggsBwonderbook bookBwonder worldB	wonder ofBwomens volleyballBwomens murderBwolfenstein tidesBwolfenstein operationBwolfenstein cyberpilotBwolf hdBwolf ageBwolcen lordsBwolcenB	woah daveBwoahBwizorbBwizards warriorsBwizards throneB
wizards ofBwizards enhancedBwizardry taleB
wizardry 8Bwizard warsBwits wagersBwitsBwithout windBwithout saladB	within inBwith tjBwith theBwith poeBwith meB
with jamieBwith friendsBwith banjokazooieBwitchspring3 refineBwitchspring3Bwitchs taleBwitchsB	witchkingBwitcheyeBwitches neoBwitcher enhancedBwitcher adventureBwitch remasteredB
witch huntBwitch academiaBwishingBwise monkeyBwiseBwisdomBwirewayB
wipeout xlBwipeout pureBwipeout pulseBwipeout omegaBwipeout fusionB
wipeout 64B	wipeout 3Bwipeout 2048B	wintersunBwinters curseBwintersBwintermoor tacticsB
wintermoorBwinter starsBwinter assaultB
winnie theBwinnieBwingspanB	wings theBwings strangerB
wings overB	wings andB
wings acesBwingmenBwingmanBwing islandBwing 2BwindwardB	windscapeBwinds leavesBwindows editionB
windows 10B	windlandsB	windforgeB	wind monkBwinback covertBwilsons heartBwilsonsBwilly unleashedBwilly morganBwilly jetmanB	wille zurBwilleB	will tellB	will rockB	will comeBwildstarBwildlife parkBwildlife adventureB
wildermythB
wild worldB	wild wildBwild westernBwild runBwild racingB	wild gunsB
wild blastBwild atB
wiki questBwikiBwik theBwikBwii pikachusB	wii musicB
wii degreeBwiggleB
wide oceanBwicked cricketBwickedBwhyd youBwhydB
who standsB
who clonedBwhiteoutB	white dogB	white dayBwhite creatureBwhistleblowerBwhirlwind overB	whirlwindBwhipseey andBwhipseeyBwhere isBwhen vikingsBwhen skiBwheels velocityBwheels beatB	whats newBwhats cookingBwhat happenedBwhackedBwestport independentBwestportBwestgateBwestern theBwestern frontBwestern adventureBwesterado doubleB	westeradoBwest theBwest onlineB	west gunsB
west frontBwerewolves withinB	were hereBwell ofB
well neverBweeping dollBweepingB
weekend inBweekendBweedcraft incB	weedcraftBweaponographistBweapon shopB	wealth ofBwealthBweaklessBwe wereBwe soarBwe meetBwe leaveBwe brawlBwayward skyBwayward manorBwayfarer ofBwayfarerBway remasteredB
way princeB	wavey theBwaveyBwavesBwaverly placeBwaverly academyBwaveformB
wave rallyBwaters antaeusBwaterloo napoleonsBwaterlooBwatchmaker 2001B
watchmakerBwatch galleryBwatch 3BwarzonesBwartileBwartech senkoBwartechBwars vrBwars trilogyB
wars superBwars spartaB
wars racerBwars obiwanBwars nightfallBwars lightsaberBwars journeyBwars futureBwars flightBwars factionsBwars eyeBwars exB	wars dualBwars definitiveB	wars daysBwars battleBwars andB	wars 2081B	wars 2009Bwarriors xtremeBwarriors volBwarriors vietnamBwarriors t72Bwarriors streetBwarriors stateBwarriors spiritBwarriors nextBwarriors katanaBwarriors joanBwarriors godseekersBwarriors dsBwarriors definitiveBwarriors codeBwarriors allstarsBwarriors ageBwarriors advanceBwarriors 4iiBwarrior viiBwarrior theBwarrior legendsBwarrior editionB	warplanesBwarpedBwarpartyBwarnings atBwarningsBwarning editionBwarlords ivBwarlord editionBwarlocks vsB
warlocks 2Bwarlock masterB	warlock 2B	warlanderBwarjetzBwarioware twistedBwarioware touchedBwarioware snappedBwarioware smoothBwarioware goldBwarioware getBwario worldBwario masterBwarheadBwarhawkBwarhammer realmBwarhammer questBwarhammer onlineBwarhammer callBwarhammer ageBwargame redBwargame europeanBwargame airlandBwarfare supremacyBwarfare sabotageBwarfare remasteredBwarfare reflexBwarfare reckoningBwarfare mobilizedBwarfare havocBwarfare ascendanceBwarfaceBward theBwarcraft wrathBwarcraft whispersBwarcraft warlordsBwarcraft shadowlandsBwarcraft mistsBwarcraft legionBwarcraft journeyBwarcraft goblinsBwarcraft curseBwarcraft classicBwarcraft cataclysmBwarcraft blackrockBwarcraft battleB	warchiefsBwarbornBwar zeroB
war winterBwar warlordBwar vrB
war vikingBwar vietnamB	war timesBwar theBwar soulstormB	war rogueBwar rockBwar redBwar pattonsB
war pattonBwar overBwar originsBwar onBwar nationsBwar machineBwar logsBwar kingdomsBwar judgmentBwar highBwar goneB	war ghostB	war frontB
war forcesB	war fleetB
war directBwar darkB
war crisisBwar condemnedB
war chainsB
war battleBwar barbarianB
war attilaBwar ascensionB	war arenaB
war arcticBwar alexanderBwar aBwanted uBwanted deadBwanted corpB
wanted 510B	wanted 50Bwanderer reloadedBwand ofBwandBwalt disneyBwaltB
walls mustBwallsBwallachia reignB	wallachiaBwallBwalkerBwakingBwakfuBwaker hdBwakeboarding hdBwailing heightsBwailingBwagersBwade hixtonsBwadeBwacky worldsBwacky worldB
wacky jobsBwBvsforceBvs zetaBvs zeonBvs tanksBvs swordBvs spyB
vs shadowsBvs segaB	vs sasukeB
vs pinballB
vs phoenixBvs pastranaB	vs ninjasBvs newBvs maxiboostBvs martiansBvs kingB	vs gnomesBvs darkdeathBvs armyB	vs afrikaB	vr worldsBvr underB	vr seriesBvr pingBvr operativeBvr kitBvr kindaBvr isleBvr helpB
vr editionBvr chroniclesBvr arenaBvoyage inspiredB
vostok incBvostokBvordermans sudokuB
vordermansBvooju islandBvoojuBvoodoo diceBvon sottendorffBvon gutB
volume iiiBvolume iBvolume 3B
voltron vrBvolleyball remixedBvolleyball championshipBvolgarr theBvolgarrBvolcano islandBvolcanoBvol 3redemptionBvol 2reminisceBvol 1rebirthB	void zeroB
void trrlmBvoid terrariumBvoezBvivietteBvitamin connectionBvitaminB	vita petsBvita editionBvisionsBvision trainingBvirus namedBvirtuaverseBvirtualon oratorioBvirtualon marzBvirtual realityBvirtual poolBvirtual arcadeBvirtua strikerBvirtua racingBvirtua beachBvirtua athleteBviral survivalBviralBviolettBviolence perfectedBvinlandB	vinci theBvince remasteredBvillainsBvilgax attacksBvilgaxBvikings attackBviking warlordBviking invasionBviirB
vii modernBvii fragmentsBvigorBvigilante 8B	vigilanteB	vigil theBvigilBviggos revengeBviggosBvietnam ultimateBvietcong purpleBvietcong fistBvietcong 2003B
vietcong 2BvideokidB	videoballB
victory vsBvictoria anBvicious circleBvice theBvi theB	vi shadesBvi riseB	vi realmsBvi gatheringB
vi advanceBvfdBvestroiaBvestaBvesselBvesperB	very veryB
very valetB	vert dirtBvertBversus ka52BverneB
vermillionBver566Bver 2Bventure originsBventure kidBventure chroniclesBvengeance andBvendetta onlineBvempireB
velocity xBvelocity ultraBvelocity bowlingB	velociboxBveliousBveilBvehicular combatB	vehicularBvehicle kitBvehicleBvegas tycoonB	vectronomBvector thrustB	vector exBvariety kitBvarietyB	variant sBvariantBvar commonsBvarBvanocore conspiracyBvanocoreBvanitiesBvanishing filesBvanguard sagaBvanguard prophecyBvaneBvampire warsBvampire starBvampire smileBvampire nightBvampire moonBvampire coastBvampire apocalypseBvalorantBvalor campaignBvalley withoutBvalkyrie driveBvalhalla hillsBvaletBvalentino rossiB	valentinoBvalentiaBvale shadowBvaleBvaldis storyBvaldisBvagrant storyBvagrantB
vae victisBvaeBvader immortalBvaderBvacation simulatorB	v8 racingBv8Bv tribesB
v outatimeBv metalBv handB	v hammersBv godsBv generationBv endBv directorsB
v championBv braveBv beyondBv arcadeB	v advanceButopiaButawarerumono zanButawarerumono preludeBusagimaru theB	usagimaruBus ultimateBus remasteredBus partBus openBus leftBus clingBuru theBuru agesBurhganBurban reignBurban jungleBurban gtBurban empireBurban danceBurban championB
upon lightBuplink hackerBuplinkBup hedgehogB
up editionBunwound futureBunwoundB
unsung warBunsungBunstoppable gorgBunstoppableBunsolved crimesBunsolvedB	unsightedBunrestB	unpluggedBuno rushBunlosing rangerBunlosingBunlimited sagaBunlimited adventureBunlimited 3Bunleashed ultimateBunleashed onBunleashed doubleBunknown plusBuniversity lifeB
universe 2Buniversalis iiBuniversalis crownBuniversal studiosBuniversal crusadeBuniversal combatBuniversal challengerB	united vrBunited soccerBunited peaceBunited offensiveBunite earthBunit editionBunit 77Bunit 4Bunit 13Bunison rebelsBunisonB
union wellBungoroBunforgottenBunforeseen incidentsB
unforeseenB	unfoldingBunfoldedBunfinished swanBunfinished businessBunexpected questB
unexpectedB	unearthedBundyingB
undrentideBundiscoveryBunderworld larasBunderworld beneathBunderworld ascendantBundertowBunderstone questB
understoneB	underrailB	undermineBunderground rivalsBunderground poolBunderdome riotB	underdomeB	underdarkBundercover theBundercover operationBunder defeatBundead knightsBundead awakeningBund boseBundBunconqueredBuncommon valorBuncommonBuncleBuncharted goldenBuncharted fightBuncharted drakesBuncharted 4Buncharted 3Buncharted 2Buncertain lightB	uncertainBuncagedBunboxedBunavowedBumurangi generationBumurangiBumiharakawaseBumbrella chroniclesBulyssesBultronB	ultratronB
ultramix 3B
ultramix 2B	ultracoreBultra vrB	ultra sunBultra smashB
ultra moonBultra minigolfB
ultra deadBultra bustamoveBultor exposedBultorB	ultimatumBultimate workoutBultimate soldierBultimate slasherBultimate sithBultimate rideBultimate quizBultimate pinballBultimate partyBultimate nesBultimate mayhemBultimate matchBultimate ghostsBultimate generalBultimate brainBultimate boxBultimate bmxBultimate blockBultimate beachBultimate battleBultimate allstarsBultima onlineBultimaBulalas cosmicBulalasBugly princeBugly americansBufo extraterrestrialsBufo aftershockBufo aftermathBufo afterlightB
ufc tapoutB
ufc suddenBudxBudraw studioBudrawBu deluxeBu actionBtyrimBtyranny bastardsBtyping chroniclesBtyphoon risingBtypesBtype 4BtypeBtycoon worldBtycoon marineBtycoon loopyB	tycoon dsBtycoon dinosaurBtycoon cityBtycoon adventuresB	tycoon 3dBtycoon 2001BtxkBtwo treasuresB	two townsBtwo sistersB
two rebelsBtwo kittiesB
two crownsBtwo aBtwisterBtwisted worldBtwisted towersBtwisted shadowBtwisted editionBtwist ofBtwins dxBtwinsBtwinbeeBtwin strikeBtwin snakesBtwin sectorBtwin breakerBtwin ageBtwelve starsBtv superstarsBtv showBtv partyBturtles turtlesBturtles smashupBturtles outB
turtles inBturtles arcadeBturtles 2003Bturtles 1989Bturtlepop journeyB	turtlepopBturtle taleBturtle adventureBturok remasteredBturok dinosaurBturok 3Bturn itBturnBturismo theBturismo sportB	turismo 6B	turismo 4B	turismo 3B	turismo 2B	turf warsBturbosBturbo turtleBturbo revivalBtunnel ratsBtunnel battleBtuningBtunes sheepBtunes racingB
tunes duckBtunes cartoonBtuner challengeBtunerBtundraBtumbly adventureBtumblyBtumblestoneB
tumbleseedB	tumble vrBtube sliderBtubeBtt superbikesBtsushima ikiBtsushima directorsBtsunami offenseBtsunami 2265B	tsum tsumBtsum festivalBtsukigimes longestB
tsukigimesBtsugunai atonementBtsugunaiBtsioqueBtrystBtruth questB
trulon theBtrulonBtrufflepigsB
true swingBtrue soldiersBtrucks racingBtrucksBtruck racingBtruck madnessBtruck apocalypseBtruberbrook aB
trrlm voidBtrrlmBtroveBtrouble witchesB
trouble inB
troubadourBtropico paradiseB	tropico 2Btroopers terranBtroopers jointBtrooper quartzB	tron runrB	troll andBtrollBtrixBtriumphBtriple deluxeB	trioncubeBtrio ofBtrioBtrinity universeBtrinity soulsBtrinity editionBtrine 3Btrilogy apprenticeBtrilogy 1941Btrillion godBtrillionBtriggerheart exelicaBtriggerheartBtrigger theBtrigger portableB
trigger 3dBtrick racingBtrick phantomBtribunalBtribes vengeanceBtribes ascendBtribes aerialBtribes 2Btriangle savingB	trials toB	trials hdBtrial trickyBtrial playgroundBtrial byB	trial andBtriadB	tri forceBtrevor chansBtrevorB
trespasserBtrend presentsBtrendBtrenchesBtrek onlineBtrek newBtrek klingonBtrek invasionBtrek encountersB
trek eliteB	trek deepB	trek awayBtrek armadaBtree ofBtree friendsBtreble encoreBtrebleBtreasures ofBtreasures extendedBtreasures deluxeBtreasure worldBtreasure troveBtreasure planetBtreasure onBtreasure ofBtreasure islandBtreasure adventuresBtreasonBtreachery inB	treacheryBtraxxpadB	traverserBtravelers 2Btrauma teamBtrash panicBtrashBtraptBtrapped deadBtrappedBtransworld snowboardingB	transposeBtransmissionB	transientBtransformers primeBtransformers decepticonsBtransformers cybertronBtransformers autobotsBtransformers animatedBtransformationBtransfixed editionB
transfixedBtranquilityB
trajectileBtraitors keepBtrainz railroadBtraining forBtrainer mathBtrainer cookingB	train simB	train setBtrain feverBtrail ofBtrailBtraffic chaosBtrafficBtrading companyBtrader riseBtrader merchantBtrade empiresBtrade destroyB
tracks theBtrackmania unitedBtrackmania sunriseBtrackmania dsBtrackmania buildBtrackmania 2003Btracker specialBtrack resurrectionB	track labBtracerB	trace nycBtrace memoryB	toycon 04B	toycon 03B	toycon 02B	toycon 01Btoybox turbosBtoyboxB
toy trialsB	toy stuntBtoy shopB
toy poodleBtoy odysseyBtoy carsBtoxic grindBtoxicB
townsmen aBtownsmenB	town heroBtower spBtower defenseBtower bloxxBtower 57Btower 3BtourystBtournament fishingBtournament editionBtournament dxBtournament championshipBtournament 2005Btournament 2003Btournament 1999Btourist trophyBtouristBtourismBtouring challengeBtouringBtour modernB	tour golfBtour editionBtour decadesBtoukiden theBtoukiden kiwamiB
toukiden 2Btouhou spellBtouhou scarletB	tough andBtoughBtouchmaster 3Btouchmaster 2BtouchedB	touch theB
touch rollBtouch myBtouch goB
touch 2018Btouch 2B
totori theBtotoriB	totemballBtotaledBtortured soulBtorturedBtortuga twoBtortugaBtorrenteBtorrentB	torna theBtornaBtormentum darkB	tormentumBtork prehistoricBtorkBtoribash violenceBtoribashBtorenBtorchB
topsyturvyBtopdeeBtopatoi spinningBtopatoiB	top tanksB	top dartsB
top anglerBtop 100B	tooth andBtoothBtoontown onlineBtoontownBtools ofBtoolsBtooki troubleBtookiB
toodee andBtoodeeB	too humanB
tony toughBtonight takeBtonelico qogaBtonelico melodyBtonelico iiBtone futureBtone colorfulBtomorrow expansionBtomorrow childrenBtomodachi lifeB	tomodachiBtomena sannerBtomenaBtomeBtomcatB
tomb kingsBtolvaBtokyo tattooB
tokyo taleBtokyo rumbleBtokyo jungleB
tokyo darkBtokyo crashB
tokyo beatBtokobot plusBtoasted sandwichBtoastedBtoadstool tourB	toadstoolBto workB
to victoryB	to ungoroBto tiaraBto thalamusBto talkBto spiesBto romeBto rideB
to revengeB
to respectBto redemptionBto raceBto popolocroisBto mysteriousBto mystBto marsBto lightspeedB	to jaburoBto infinityBto indiaBto honorBto hellB	to heavenBto guangdongBto growBto goBto girlsB
to gehennaBto fifaBto fiddlersBto endBto earthBto dreamBto deadrockB
to calvaryB	to berlinBto bedBto beBto batuuBto ballhallaBto armsB	to arkhamBto aB
tnt racersBtntB	tj lavinsBtj cloutierBtitanfall expeditionBtitan humanityBtitan attacksBtiqalBtiny troopersB	tiny traxB
tiny tinasB
tiny racerB
tiny clawsBtiny brainsBtiny bigBtiny barbarianB	tiny bangBtinas assaultBtinasBtimestone piratesB	timestoneBtimes crosswordsBtimelineBtimelieBtime wandererBtime vrBtime ufoBtime sweeperB
time spaceBtime reshelledBtime recoilB
time pilotBtime ofBtime masterBtime hollowBtime heyBtime gentlemenBtime funB	time finnBtime explosionBtime commandoBtime carnageBtime autobahnBtime aceBtill youBtill theB	tigre theBtigreBtightB	tierkreisBticktock travelersBticktockB	ticket toBticketBtibetBtiberian twilightBtiberianBtiara iiBtiaraBthunderstrike operationBthunderstrikeBthunderflashBthundercatsBthunderbirdsBthunder tanksBthunder hurricaneBthunder bladeBthrustB	throwbackBthrones patriotsB
thrones ofBthrones genesisBthrone battleB
threepwoodBthree stoogesBthree musketeersBthreatBthousandyear doorBthousandyearBthothBthornsBthornBthirty flightsBthirtyB
third waveB	third theBthird lightningBthird genkiB
third dawnBthird brightBthingysB	things onBthin silenceBthinBthievius raccoonusBthieviusBthieves seasonB
thiefs endBthiefsBthief ofBthief iiBthief ancestryBthief aB
they stoleB
they shallBthey lieB	they cameB
they bleedBthere omegaB
there cameBtheme parksBthehunter callB	thehunterB
theater ofB	the yorhaB	the yawhgB	the yagerBthe xboxB
the writerBthe woundedBthe wookieesB
the wolvesB
the wizardBthe witchkingBthe wiseBthe westportBthe westernBthe weaponographistB	the wavesBthe watchmakerBthe wastelandBthe wardBthe warchiefsBthe wandBthe wallBthe vikingsB
the vikingBthe videokidBthe vanocoreBthe vanitiesBthe vanguardB
the valleyBthe valeBthe unwoundB
the unsungBthe unknownBthe universalBthe unfinishedBthe unexpectedBthe underworldBthe underdarkBthe uncertainBthe umbrellaBthe uglyBthe twinB
the twelveBthe turtlesB
the turtleBthe tsunamiB	the trialB	the triadBthe trenchesB	the trainB	the trailB
the tracksBthe tourystBthe tormentedBthe topBthe tomorrowBthe toastedB	the titanBthe tinyBthe timestoneBthe timeB
the throneB
the threatBthe thousandyearBthe thinBthe thieviusBthe textorcistB
the tengusBthe taxidermistBthe taleB	the swarmBthe superheroBthe sunstoneBthe stretchersB	the stingBthe stillnessBthe stetchkovBthe stealthBthe starshipB
the starryBthe stanleyBthe stairwayBthe spyB	the sporeBthe splattersBthe spartansB	the southBthe soullessBthe soulBthe sorcererBthe sopranosBthe solitaireB	the sochiBthe smokingBthe slyB
the sirensBthe sinistralsBthe singularityBthe signifierBthe showdownB	the shootBthe shipBthe shinsengumiBthe shatteringB
the seriesB
the sequelB	the seedsBthe scourgeBthe scorpioB	the scarsBthe saiyansBthe sagaBthe rubB
the rocketBthe roadBthe rippingBthe righteousBthe riftBthe rhynocsBthe rhombusBthe revengeBthe revenantsBthe resurrectionBthe resistanceBthe recruitBthe reconstructionB	the rebelB
the realmsBthe realB
the ravingB
the ransomB
the randomBthe rampBthe rainbowB
the rabbitB	the quietBthe puzzlingBthe pursuitBthe processionBthe princessBthe primordialsBthe precursorBthe powerpuffBthe portalsB	the poohsB	the plumeB
the planetB
the planesB
the plagueBthe pitBthe pharaohsB
the perilsBthe penguinsBthe peanutsBthe patriotsBthe passiveBthe partnersBthe painfulB	the padreBthe overworldBthe overlordB
the outfitBthe origamiB
the orientB
the oracleBthe olympicsB	the oceanBthe obraBthe oblivionBthe novelistB
the nonaryBthe niohB	the nexusB
the nathanBthe mythBthe munchablesBthe moveBthe moosemanB	the mooilBthe monstersBthe monophonicBthe monkeysB	the moneyB
the momentBthe moleB
the modernBthe mobBthe miracleB
the minishB	the minisB	the milkyB
the middleB	the metalBthe mercenariesB
the menhirBthe melodiasBthe megaB	the medesBthe meatballBthe mazeB	the mayanBthe mawBthe masterplanB
the maskedBthe maskBthe marvellousBthe markBthe malgraveBthe magnificentBthe magisterBthe mageBthe macrossBthe machineB	the lupusBthe luftwaffeBthe lowB	the lotusB	the looseB	the limitBthe lifeBthe lichBthe liandriBthe leviathanB	the landsBthe lairBthe labyrinthBthe koriodanBthe koreB
the knightBthe kinnikumanBthe killingBthe keyBthe keepB	the kasaiBthe karakuriBthe jupiterBthe journalB
the jaggedBthe jadeBthe jackassBthe isleBthe invincibleBthe inventoryBthe interactiveB
the insultBthe inquisitionBthe inpatientBthe infinityBthe infernalBthe infectiousB
the infamyBthe infamousB
the ilvardBthe icoB
the hustleBthe huntsmanB
the hunterBthe hugeBthe homelandBthe hollywoodBthe hiveBthe highBthe hermudaBthe heavenlyBthe heartlandBthe hawkBthe hatB	the hanseBthe hangmanBthe gunstringerB
the guidedB	the guestB
the grudgeBthe grooveridersBthe gridB	the grandBthe godslayerBthe gladiatorsBthe gladiatorBthe girlB	the giantBthe ghouliesB
the ghostsB	the genosB	the genieBthe gateBthe galacticB	the furonB	the fungiBthe fullBthe frontierBthe friendsB
the fridayB	the franzB	the frameBthe fourB
the flyingB
the flowerB	the fleetBthe flattestB	the flashBthe fittestBthe fistBthe fishBthe fireflyBthe fireB
the finestB	the filesBthe fidelioBthe fathersBthe fantasyBthe fantasticBthe fanBthe falseboundB	the falseB
the falconB
the fafnirBthe faceB	the fableBthe eyesBthe experimentBthe executionerBthe everlastingBthe etherealBthe eternityBthe entertainmentBthe encounterBthe emperorB	the elvesBthe ellyBthe elementsBthe egyptianBthe echoBthe easternBthe eastBthe earthlingsB
the eaglesBthe dutchmanB	the dummyBthe duelistsBthe dubBthe drownedBthe draculaBthe donsBthe dominatrixBthe dominatorsB	the dollsBthe dogsBthe doggB
the doctorBthe dnaB
the divineBthe diabolicalB	the devilB
the detailBthe destinyBthe destinedBthe descendantBthe delusionsBthe definitiveBthe deerBthe deathfinBthe daysBthe dawnBthe darkhulBthe darkagesB	the danceBthe curiousB	the cubesBthe crusadersB	the crowsBthe creatureBthe creatorB
the cortexBthe corporateB	the coralBthe consequenceBthe conquerorsBthe conduitBthe conBthe colonistsBthe colliderB	the clownB
the cloudsB
the clonesBthe clockworkBthe charnelBthe chaoticBthe championsB
the centerBthe celestialBthe caseBthe carnivalBthe capitalBthe calmBthe cainB
the bulletBthe bugBthe britishB
the brightBthe braceletsBthe bouncerB
the bottleB	the bogeyBthe blueBthe biggestBthe bigfootBthe betrayerBthe betrayalB
the belkanBthe beholderBthe beginnersBthe beastmenBthe battleboxBthe bandB
the balladBthe ballBthe badlandsB	the babesB	the azranB	the azothB	the atticBthe atlanticBthe asskickersBthe assignmentBthe assassinB	the asianBthe ascensionBthe artifactsB	the arnorB	the argesB
the arcaneBthe aquaticBthe antsB	the angryBthe andosiaBthe ancientsBthe ampBthe americasB	the alleyBthe agesB	the afterBthe adventurerBthe acrobatBthe academyB	the abbeyBthe 99Bthe 9Bthe 8thBthe 80sBthe 5thBthe 4Bthe 2dBthe 25thBthe 1stB	thats youBthatsBthat matterBthat dragonB
that daresB
than lightBthan 6BthalamusBth3 planBth3Btextorcist theB
textorcistB
tex murphyBtexBtetris splashBtetris evolutionB	tetris dsBtetris axisB	tetris 99BtetheredBtests dsBtestsB	teslapunkBtesla effectBterroverBterror squidB	terror inBterrawars newB	terrawarsB	terrariumBterran conflictBterran ascendancyBterra aBterraBterminationBteraB	tentaclesBtensei nocturneBtensBtenormans revengeB	tenormansBtennis ultraBtennis powerBtennis openB	tennis inB
tennis getBtennis acesB
tennis 2k2Btenkaichi tagBtenkai knightsBtenkaiBtengus discipleBtengusBtengamiBtenchu zBtenchu wrathBtenchu stealthBtenchu returnBtenchu fatalBtenchu darkBtenchu 2Bten kateB	temple iiBtempest pirateBtelltale miniseriesBteleglitch dieB
teleglitchBtekken revolutionBtekken hybridBtekken darkBtekken advanceBtekken 4B	tekken 3dBtekken 3Btekken 2Bteenage zombiesB
teen powerBteen hungerBtee itBtee 2Btecmo classicBtechnobabylonBtechnicaBtechnic beatBtechnicBtech supportBtechBtears toBtearaway unfoldedBtear ofBtearBteam swBteam protomanB	team ogreBteam fortressBteam dxBteam colonelB
team arenaBtdr 2000BtdrBtaxidermistB	taxi fareB
taxi catchBtaxi 3Btaxi 2Btattoo girlsBtattooBtatsunoko vsB	tatsunokoBtatsujin rhythmicBtasukete takosanBtasuketeBtastee lethalBtasteeBtasosBtasomachi behindB	tasomachiBtarzan returnBtartarusBtarr chroniclesBtarrBtarget terrorBtarget libertyB
tappingo 2BtapperBtapout 2BtapBtaos adventureBtaosBtao fengBtanuki justiceBtanukiBtannenB
tanks xboxBtank universalBtank troopersBtank commanderBtank commandB	tank beatBtank battlesBtangram ver566BtangramB
tangledeepBtangle towerBtangleB	tang tangBtamarinBtamagotchi partyBtalk toB	tales theBtales critterBtales ancientBtale ivBtakosanBtako tasuketeBtako definitiveBtakedown redBtakedaBtake commandB	take backBtaishiB	taisen ogBtaipanBtainted grailBtainted bloodlinesBtail ofB
taiko drumBtaiketsuBtaikaiBtadpole trebleBtadpoleB
tactics hdB
tactics dsBtactics clubBtactics brotherhoodBtactics advanceB
tactics a2Btactical strikeBtactical opsBtactical interventionBtactical commandBtactical combatBtabula rasaBtabulaB
table miniBtable mannersBt72 tankBt72Bsystem riftBsynthetik ultimateB	synthetikBsynthesizerBsynth ridersBsynthBsyndicate jackBsyndicate 2012Bsynapse primeB	synapse 2Bsymphonia dawnBsymphonia chroniclesBsympathyBsymmetry theBsymmetry 2018Bsymbols adventureBsymbolsBsymBsylpheed arcBsylpheedBsx superstarBsxB
swords theBswords anniversaryBswords adventuresBsword sworceryBsword hdB
sword gameBsword destinyBsword coastBsword 3Bsworcery epBsworceryBswitch stanceBswingerz golfBswingerzB
swing awayBswineBswim outBswimB
sweet shopB
sweet homeBsweeperBswat targetBswat 3B	swapquestBswanBswampys revengeBswampysB	swamps ofBswampsBswB	svc chaosBsvcB	suzuki ttBsuzukiBsuzerainBsushidoBsushi strikerBsushiB	survivorsBsurvivor storiesBsurvivor seriesBsurvivor overclockedBsurvivor 2001Bsurvive stormB	survive 2Bsurvival journeyBsurvival gloriaBsurvival editionBsurgical unitBsurgicalBsurge deluxeBsurfing h3oBsurface thunderBsurfaceB
surf worldBsurf rocketBsurf nBsupreme leagueBsupport errorBsupportBsupersweet pinballB
supersweetBsuperstars v8Bsuperstar baseballBsupersonic acrobaticBsuperslime editionB
superslimeBsuperpower 2Bsupernova 2BsupernaturalBsupermini festaB	superminiBsuperman theBsuperior firepowerBsuperiorBsuperhypercubeB	superheroBsuperfrog hdB	superfrogBsuperepic theB	superepicBsuperdimension neptuneBsuperdimensionBsupercross encoreBsuperbrothers swordBsuperbrothersB
superbikesBsuperbike 2001Bsuper ultraBsuper trucksB	super toyB
super starBsuper snazzyBsuper sluggersBsuper seducerBsuper scribblenautsB
super rushBsuper runaboutB
super rudeB	super rubBsuper princessBsuper partyBsuper paperB	super oneBsuper mysteryBsuper motherloadBsuper mondayBsuper megaforceBsuper luigiBsuper littleB	super joyBsuper ghoulsBsuper galaxyBsuper explodingBsuper dodgeballBsuper dodgeBsuper deluxeBsuper crushBsuper contraBsuper collapseBsuper circuitBsuper chariotBsuper challengeBsuper bubbleBsuper bombadBsuper bloodB
super beatB
super armyBsunstone odysseyBsunstoneBsunshine islandsBsunsBsunriseBsunny garciaBsunnyBsunless skiesBsunless seaBsundered eldritchBsunageBsun theB
sun risingBsun isBsun darkBsummoner soulBsummoner raidouB
summoner aBsummit strikeBsummer sportsBsummer heatBsumireBsumioni demonBsumioniBsuit leynosBsuit infinityB
suikoden vBsuikoden tierkreisBsuikoden tacticsBsuikoden ivBsuikoden iiiBsuikoden iiBsuicide guyBsuguri xeditionBsuguriBsuffering ofBsudoku gridmasterBsudekiBsudden impactBsudden deathB	successorBsubsurface circularB
subsurfaceBsubsistenceB	submersedBsubmarine titansB	submarineBsubject 2923B
subject 13Bsub warsBsub rebellionBstyling starBstylingBstyle rotozoaBstyle rotohexBstyle pictobitsBstyle orbientBstyle lightBstyle cubelloB
style baseBstyle aquiaBstygian reignBstygianBsturmovik forgottenBsturmovik cliffsBsturmovik battleBsturmovik 1946Bstupid invadersBstupidBstunts effectsBstunt gpBstunt driverB
stunt bikeB
stuff packBstuffBstudios themeBstudiosBstudent allianceBstudentB
strugglingBstructBstronghold warlordsBstronghold legendsBstronghold 3Bstronghold 2Bstrobophagia raveBstrobophagiaBstrikers edgeBstrikers chargedBstriker theBstriker packBstriker 2002B
strike theBstrike operationB	strike iiBstrike 3B	strider 2B	stretchmoB
stretchersBstretch panicBstretchBstress aBstressBstrength ofBstrengthBstreet vertBstreet traceBstreet supremacyBstreet showdownBstreet racerBstreet powerBstreet onceB
street jamBstreet brawlBstreet basketballBstrategic commandB	strategicBstraniaBstrangelandBstranding directorsBstranded sailsBstranded deepBstrafeBstory revolutionBstory racerB	story newBstory maniaBstory completeBstory cinderedBstory bowserBstory abyssalBstory aboutBstory 3dB
stormreachB	stormlandBstormersBstormbreakerBstorm warningBstorm trilogyBstorm reloadedBstorm ofBstorm legionBstorm islandBstorm groundBstorm frontlineBstorm farewellBstorm empireBstorm alliesBstories fromBstop stressBstop sneakinBstoogesBstones wildfireBstones thunderflashBstone ofBstone magicBstone collectionBstone 2Bstolen kingdomBstolen 2005Bstole myB
stole maxsB
stoked bigB	stock carBstockB
stitchy inBstitchyBstitch experimentBstirring abyssBstirringBstingBstimulus packageBstimulusBstillness ofB	stillnessBstill thereBstill aliveBstikbold dodgeballB
stikbold aBstifledBsticker starBstickerBstewBstetchkov syndicateB	stetchkovBsteredenn binaryBstephens sausageBstephensB	step rollBstellaris utopiaBstellaris apocalypseBstellar editionBstellarBstella glowBstella deusB
steep roadB	steeltownBsteel soldiersB	steel skyBsteel lancerBsteel horizonBsteel empireBsteel beastsBsteel assaultBsteel 2Bsteamworld questBsteamworks andB
steamworksBstealth bastardBsteal princessB	steal ourBsteadyBstation siliconBstation santaBstatikBstatesBstasisBstarwhalBstarve giantBstarve consoleBstarting lifeBstarting fiveBstarsignB	starshipsBstarship defenseBstarship damreyBstarshatterBstarseed pilgrimBstarseedB	stars theBstars poweredBstars inBstars iiBstars bravoB
stars bornBstars alphaBstarry skiesBstarryBstarring goemonBstarring daffyB	starlightBstarhawkBstarfyBstarfighter specialBstardust portableBstardust hdBstardust deltaBstardust acceleratorBstardrone extremeBstardrive 2Bstarcraft remasteredBstarcraft 64B	starboundBstarblox incBstarbloxBstarblood arenaB	starbloodBstarbase commanderBstarbaseB	star zeroBstar xB
star ultraBstar successorB
star storyBstar soldierB	star sagaB	star rushBstar renegadesBstar raidersB	star lostBstar horizonBstar harmonyBstar hammerBstar controlBstar conflictBstar collectionBstar alliesBstanley parableBstanleyBstands behindBstandsBstandoffBstanceBstalker shadowBstalker clearBstalker callB	stalin vsBstalinBstakesBstake fortuneBstakeBstage iiB	stadium 2Bstacking theBssx blurB	squishiesBsquigglepants 3dBsquidBsqueeballs partyB
squeeballsBsqueak squadBsqueakBsquarepants theBsquarepants lightsBsquarepants editionBsquared mindBsquadron wwiiBsquadron iiiBsquadron iiB	squad theBsquad operationBsquad nemesisBsquad leaderBsquad commandBsquad 2B
spyro yearBspyro shadowBspyro seasonBspyro orangeBspyro attackBspyro 2Bspyglass boardBspyglassBspyborgsBspy vsBspy inBspy fictionBspy chameleonBspy andBspuds unearthedBspudsBsprungBsprinting wolfB	sprintingBspring breakBspringBsprayBsprach zarathustraBsprachB	spotlightBsportsfriendsBsportsbarvrB	sports tjB
sports theBsports tennisBsports superstarsBsports soccerBsports snowboardingBsports skydivingBsports showdownBsports seasonBsports rivalsBsports resortBsports pureBsports partyBsports paradiseBsports motocrossB
sports mixB
sports jamBsports freedomBsports footballBsports fishingB	sports dsBsports connectionBsports clubBsports 3B
spore warsBspore galacticBspore creaturesBspore creatureBsplitzB	splattersBsplashdown ridesBspitfire heroesBspitfireBspirits sprintingBspirits spellsBspirits soaringB	spirits 2Bspirit tracksBspirit dimensionsBspirit detectiveBspirit dancerBspirit cameraBspirit callerBspintires gameBspinning throughBspinningBspinchBspinachBspin theB
spin cycleBspikesBspikers virtuaBspikersBspikeout battleBspikeoutB	spike proBspiffingBspiesBspidyBspiderman turfBspiderman silverBspiderman mysteriosBspiderman battleB	sphinx anBsphinx 2B	spheroidsBsphere leifthrasirBspelunker partyBspelunker hdB
spellspireBspellforce theBspell bubbleBspellBspeer editionBspeerB	speedzoneBspeedway usaBspeedwayBspeedrunnersB
speed zoneBspeed worldBspeed porscheB
speed highBspeed devilsBspeed championsBspectrobes originsBspectrobes beyondBspectral soulsBspectra 8bitBspectra 2015BspectersB
specter ofBspecial operationsBspecial episodeB	specforceB	spearheadBspear blackBspeaking simulatorBspeakingBspawn inBspartansBspartan strikeBspartacus legendsB	spartacusBsparcBspanish coachBspanishBspacetime continuumBspaceport janitorB	spaceportBspacecomB	spacechemBspaceball revolutionB	spaceballBspace weBspace traderBspace stationBspace siegeBspace shooterBspace rangersBspace raidersBspace piratesBspace oddityB
space nineBspace mutantBspace interceptorBspace horseB
space hackBspace giraffeBspace forceBspace chimpsBspace bustamoveBspace battlesB	space arkBspace adventuresBspace aboutB
space 2008Bsp 2BsowlsBsoviet assaultBsovietB
sovereignsBsouth pacificBsoulsilver versionB
soulsilverBsouls resurrectionBsouls prepareBsouls ofBsouls collectionBsouls artoriasBsoulless armyBsoullessBsoulcalibur lostBsoulcalibur legendsBsoulcalibur iiiBsoulcalibur brokenBsoulbringerBsoulblighterB
soulblightBsoul resurreccionBsoul ofB
soul nomadBsoul iiBsoul harvestBsoul hackersBsoul bubblesBsottendorff andBsottendorffBsorry slidersBsorcery sagaBsorcery partsBsopranos roadBsopranosB
sophie theBsophieBsonic shuffleBsonic pinballBsonic knucklesB
sonic gemsB
sonic freeBsonic classicBsonic chroniclesBsonic battleB	songs andBsongsBsongbird symphonyBsongbirdBsometimes monstersB	sometimesBsomeday youllBsomedayBsombreBsoltrio solitaireBsoltrioBsolstice chroniclesB
solstice 2Bsolo islandsBsolitude theBsolitaire conspiracyB	solid theBsolid snakeBsolid peaceBsolid digitalBsolid 4Bsoldnerx himmelssturmerBsoldner secretBsoldnerBsoldiers soulBsoldiers ofBsoldiers heroesBsoldiers hdBsoldiers coldB	soldier rBsoldier eliteBsoldier bloodBsolatorobo redB
solatoroboBsolasta crownBsolastaB	solas 128BsolasBsolarixBsolar shifterB	solar boyBsolar 2Bsol survivorB
sol exodusBsokobondBsoireeBsoda drinkerBsodaBsocom iiBsocom 4Bsocom 3BsocietyB	societiesB
sochi 2014BsochiBsoccer maniaBsoccer 2Bsoaring hawkBsoaringBsoarBsoakedBso manyBsnowy escapeBsnowyBsnowfallBsnowboarding worldBsnowboarding roadBsnowboard kidsB
snow placeBsnowBsnoopys grandBsnoopysBsnoopy flyingBsnk proBsnk galsBsnipperclips plusBsnipperclips cutBsniper challengeB
sniper artBsneakinBsneakersB
sneak kingBsnazzy editionBsnazzyBsnapshot 2012BsnapshotBsnappedBsnakesBsnack worldBsnackBsmooth movesBsmoothBsmoking mirrorBsmokingBsmileyBsmileBsmelterBsmashupBsmasherBsmashbox arenaBsmashboxBsmash tvBsmash nB	smash hitB
smash crewBsmarty pantsBsmartyB
smart bombBsmart asBsmall brawlB
small armsBsmackdown shutBsmackdown justBsmackdown hereBsmackdown 2Bsly collectionBsly 3Bsly 2BsluggersBslug xBslug advanceBslug 7Bslot carBslotBslimesan superslimeBslime rancherBslight caseBslightBslidersBsliderBsleuth hackersBsleuth completeBsleepsBsleeping kingBsleep tightBsleep ofB
sled stormBsledBslayer wrathBslayer editionBslayaway campBslayawayBslasher switchBslasherBslashBslapshotB	slam boltB
slai steelBslaiBskytree villageBskytreeBskyshines bedlamB	skyshinesBskyrim hearthfireBskyrim dragonbornBskylines xboxBskylines snowfallBskylines playstationBskylines nintendoBskylines naturalBskylines afterBskyhillB	skygunnerB	skydivingBskydive proximityBskydiveBsky theBsky scBsky odysseyBsky invasionBsky infinityBsky fortressBsky derelictsBsky crawlersBsky childrenB
sky beyondB	skulls ofBskullsBskullBskul theBskulBskies iiB
skies highBskies eliteBski snowboardB	ski liftsBsketcherBskelter nightmaresB	skelter 2BskellboyB
skelattackB	skater 2xBskateboarding featuringBsixty secondBsixtyB	six flagsBsix criticalB
six covertBsituation comedyB	situationB
sith lordsBsith editionBsisters underBsisters royaleBsisters generationBsisters dreamBsirens callBsiren bloodBsir youB
sir branteBsins knightsBsinking islandB
sinistralsBsingstar volBsingstar ultimateBsingstar rocksBsingstar legendsBsingstar danceBsingstar countryBsingstar celebrationBsingstar ampedBsingstar 90sBsingstar 80sBsingstar 2008Bsingles flirtBsinglesB
sing queenB
sing partyBsince januaryBsinceBsin punishmentBsin episodesBsimulator nintendoBsimulator initiateBsimulator experienceBsimulator cprBsimulator anniversaryBsimulator 3Bsimulator 2012Bsimulator 2008Bsimulator 2006Bsimulator 2002Bsimulator 20Bsimulator 18Bsimulator 16Bsimulacra 2Bsims vacationBsims unleashedBsims superstarBsims petBsims onlineB
sims makinB
sims livinB	sims lifeB
sims houseBsims hotBsims castawayB	simracingBsimpsons wrestlingBsimpsons skateboardingBsimgolfB
simcoasterBsimcity societiesB
simcity dsBsimcity citiesBsimcity 3000Bsimcity 2000Bsimanimals africaB	silvergunB
silverfallBsilver liningBsilver earringBsilver eagleBsilver chainsBsilpheed theBsilpheedBsilmeriaBsilicon valleyBsilicon dreamsBsilhouette mirageBsilent stormBsilent oathBsilent lineB
silence ofB	signifierB	signatureB
signal opsBsignal fromBsign ofBsigmar stormBsigmarB
sigma starB
sigma plusBsigil bladeBsigilBsight vietnamBsiesta fiestaBsiestaBsiegecraft commanderB
siegecraftBsiege throneBsiege survivalBsiege legendsBsideway newBsidewayBsidetrackedBsidescrollerB
side storyBsickleBsiaB	shut yourBshutBshuggyBshuffle dungeonBshrouded islesBshroudedBshroudB
shreknrollBshrek superBshrek hassleBshred nebulaBshoxBshowtime championshipBshowdown effectBshowcaseBshow ofBshow mordecaiB	show kingBshow 20Bshow 19Bshow 18Bshow 17Bshow 16Bshotest shogiBshotestBshot onlineBshort peaceB
shops taleBshopsBshop deB	shop chopBshop 2Bshooty fruityBshootyBshootout 2004Bshootout 2003Bshootout 2002Bshootmania stormB
shootmaniaBshooter vengeanceBshooter ultimateBshooter theBshooter primeBshooter plantBshooter forB	shooter 2B
shoot manyBshogo mobileBshogoBshogiB	shodown vBshodown specialBshodown senB
shodown iiBshock troopersBshock forceBshock enhancedBshock 2BshirinBshipwreckedBshipwreck showdownB	shipwreckBshippuden shinobiBshippuden narutoBshippuden legendsBshippuden kizunaBshippuden dragonBshippuden clashBshioBshinyBshinsengumiBshinsekai intoB	shinsekaiBshinobido 2B	shinobidoBshinobi rumbleBshining tearsBshining battleBshikigami iiiB	shikigamiBshiftlings enhancedBshifting worldBshiftingBshiftersB
shifter exBshifterBshift happensBshift extendedBshift extendBshield dualBshibuya scrambleBshibuyaBsherwoodBsheltered 2Bshelter generationsB	shelter 2Bshell enhancedBsheep raiderBshawarmageddonB
shatteringBshattered taleBshattered soldierBshattered landsBshattered horizonBshattered galaxyBshattered crystalBshattered bladeBshatter 2009B	share theBshareB
shardlightBshapes beatsBshallie plusBshallie alchemistsB	shall notBshallBshaking theBshakingBshake itBshakeB
shafers atBshafersBshadowverse championsBshadowverseBshadows hereticBshadowrun returnsBshadowrun hongBshadowrun dragonfallBshadowrun chroniclesBshadowgrounds survivorB
shadowgateBshadowflareB
shadowbaneBshadow ultimateBshadow torchBshadow puppeteerBshadow planetBshadow magicBshadow legacyBshadow kingBshadow harvestBshadow engineBshadow dustB
shadow bugBshadow bladeBshadow assaultBshadoB	shades ofBshadesBshade wrathBshadeBsgzh schoolBsgzhBsevered steelBsever iiBseven watersB	seven volB	seven theBseven shipsBseven samuraiBseven knightsBseven kingdomsBseven godsendsBseven gamesBseven deadlyBseven acolytesB	seuss theBseussBsettlers riseBsettlers historyB
settlers 7BsetarrifB	set matchBset gameB
sessions 2BsessionBsesame streetBsesameBservedBservants ofBservantsBserious samsBseries zeldaBseries terminationBseries superBseries pacmanBseries metroidB
series iceBseries excitebikeB	series drBseries donkeyBseries deluxeBseries crashdownBseries castlevaniaBseries 2005Bseries 2003Bsera islandBseraB	sequel toBsequelBsepterra coreBsepterraBsentinels ofBsentinels aegisBsentinel descendantsBsensible worldBsensibleBsenseyBsense aBsenseBsenryaku viiBsenryakuB	senki theBsenkiBsenB	semper fiBsemperBsempai legendsBsempaiBselfBselection volBselection 2Bseiya soldiersBseiya sanctuaryBseiya braveBsega sportsBsega marineB	sega hardBsega classicsBsega casinoBsega arcadeBsega 3dBseeking dawnBseekingBseek andBseed ofB
seed neverBseed battleBsee theBseeBseducer howBseducerB	seduce meBseduceBsecret saturdaysBsecret ringsBsecret ponchosBsecret pathsBsecret libraryB	secret atBsecond waveBsecond storyB
second sonBsecond shooterBsecond opinionBsecond evolutionB
second endBseasons trioBseasons pioneersBseasons friendsBseasons fairytaleBseason 3Bseason 2015Bseason 2014Bseason 2013Bseason 2012Bseason 2008BseamanBseals tacticalBseals confrontationBseals combinedBseabladeB
sea traderBsea lifeB	sea gatesBsea dogsBsea 2Bscurge hiveBscurgeBscrolls legendsBscrolls bladesBscrewjumperB
screamrideBscream teamBscream arenaB	scratchesB	scrappersBscrap metalBscrap gardenBscram kittyBscramBscrabbleBscourge projectBscourge outbreakBscorpio ritualBscorpioB
score rushBscoreBscorched earthBscorchedBscope completeBscope 3Bscope 2BscooterBscoobydoo whosB
scooby dooBscoobyBscifi postapocalypticBscifiBscience fairBschool girlBschizoidBsceneryB	scavengerBscars ofBscarsBscarlet graceBscarlet curiosityBscariest policeBscariestBscarface moneyBscanner sombreBscannerBscale racingBscaleBscBsbk09 superbikeBsbk09Bsbk07 superbikeBsbk07Bsbk xBsbk snowboardBsbk generationsBsaysBsayonara umiharakawaseBsawyers locomotionBsawyersB	saw blackBsavvy stylingBsavvy fashionB
saviors ofB
saving theBsavingB	save kochB
savage theBsavage skiesBsavage moonBsavage 2Bsausage rollBsausageBsaurianBsaturdays beastsB	saturdaysBsaturday morningBsaturdayBsatinavBsatellite reignB	satelliteBsasukeBsapphire wingsBsaplingBsantaBsannerBsanitys requiemBsanitysBsanity aikensBsanityBsangfroid talesB	sangfroidBsandwich ofBsandwichB	sandstormBsanctus reachBsanctusBsanctum 2011Bsanctuary battleBsanadaBsamus returnsBsamusBsamurais destinyBsamuraisBsamurai westernBsamurai swordBsamurai storyBsamurai squadBsamurai legendBsamurai gunnBsamurai champlooBsamurai artBsamurai 20xxB
sams bogusBsamsB
samorost 3BsamorostBsame stitchBsameBsam advanceBsam 4B
salary manBsalaryBsalad amuseboucheBsaladBsakura samuraiBsaiyuki journeyBsaiyukiBsaiyansBsails explorersB	saga troyBsaga thronesBsaga theBsaga scarletBsaga laevateinBsaga kororinpaB
saga finalBsaga endlessBsaga chroniclesBsaga bowsersBsaga aBsaga 1Bsafari animalBsadameBsacrifice deltaBsacrifice 2000Bsacred underworldBsacred tomeBsacred symbolsBsacred stonesBsacred ringsBsacred cardsBsacred bloodBsackboys prehistoricBsackboysB
sabre wulfBsabre squadronBsabotageBs challengeBryzomBrymdresaBryl pathBrylBryder whiteBryderBrxB
rwby grimmBrwbyBrusty trailsBrustyBrustin parrBrustinBrussia bleedsBrush remasteredBrush ofB	rush hourBrush extendedB	rush brosBrush adventureB	rush 2049Brush 2BrunrBrunning backwardsBrunner marsBrunner legacyBrungunjumpgunB	runeterraBrunespell overtureB	runespellBrunes ofBrunesBrunersBrune vikingBrune iiB
rune hallsB	rune 2000Brunaway theBrunabout sanBrunaboutBrun warzonesBrun theBrun kingB
run galaxyBrun byBrun 2BrumuBrummyBrumbly tumblyBrumblyBrumble worldBrumble uBrumble revolutionBrumble racingBrumble blastB
ruler coldB
ruler 2020B
ruler 2010Brule ofBruleBrugby challengeBrugby 18Brugby 08Brugby 06Bruff triggerBruffBrueB	rude bearBrudeBruby versionBruby sapphireBrubiks puzzleBrub rabbitsBrub aB	rtype iiiBrtype commandBrtx redBrtxB
rpm tuningBrpmBrpg postapocalypticBroyale fiveBroyale 2Broyal rumbleBroyal libraryB	royal airBrowans battleBrowansBroving rogueBrovingBroverBroute 66BroundupB
round golfBrotozoaBrotohexBrotating octopusBrotatingBrotasticB	rossi theBrossiBroses xxBrose valkyrieBrose inB
rorona theBroot letterBroot doubleB	root beerBroom toBroom mayhemBroom 215Broogoo twistedBroogoo attackBroninBrondo bulletBronde 2Brome vaeBrome risingBrome remasteredBromanumBromantic distancingBromanticBromance dawnBromanaBrolling thunderBrollerzB
rollers ofBrollersBroller rescueBroller coasterBrollcage stageBrollcageBroll racingBroleBrokiB	roguebookBrogue wizardsBrogue universeBrogue toBrogue stormersBrogue statesBrogue pilotB	rogue oneBrogue lordsBrogue leaderBrogue heroesBrogue galaxyBrodgers radicalBrocky balboaB	rocky andB	rocks theBrocketpowered battlecarsBrocketpoweredB
rocketbowlBrocketbirds 2Brocket slimeBrocket riotBrocket racersB	rocket iiB	rock tourBrock shooterBrock nBrock managerBrock legendsBrock galacticBrock angelzBrobotron 2084BrobotronBrobotech theBrobot namedB	robot kitB
robot golfBrobot arenaBrobot alchemicB	robonautsBrobojamBrobocraft infinityB	robocraftBrobocopBrobocalypse beaverBrobobotBrobo recallB
robo arenaBrobinson theBrobinsonBroasted mothsBroastedBroar primalBroar extremeBroar 4Broar 3Broads toB	road rashBroad onBroad ofBroad adventureBroachBrmxBriveraB	riven theBrivenBrive ultimateBrivals 2004Brivals 2Britual crownBrisk systemBrisk globalBrisk factionsBrising mythsBrising chopB	rise raceBrise fantasiaB	rise fallBrise andBriptos rampageBriptosB
riptide gpBripping friendsBrippingBripped 1995BrippedBrings tacticsBringo ishikawaBringoBring fitBrimworldBrims racingBrimsBrigs mechanizedBrigsBrignrollBrights reckoningB	righteousBright brainBrightBrigby inBrigbyBrigBriftstar raidersBriftstarB
rift stormB
rift apartBriding spiritsBriding hoodsB
rides goneBridesBriders zeroBrider stormbreakerB
ride turboBricoBrichard aliceBrichardBrhythms acrossBrhythmsBrhythmic adventureBrhythmicBrhythm thiefBrhythm fighterBrhythm exerciseBrhythm danceBrhynocsB
rhombus ofBrhombusBrhodan adventureBrhodanBrhiannon curseBrhiannonBrhem 3B
rhapsody aBrhapsodyB	rf onlineBrfBrezurrectionBrezero startingBrezeroBrez hdBrexBrevolution marioBrevolution konamixBrevolution iiiBrevolution directorsBrevolution countryBrevolution 2BrevoltBrevival editionBreverseBreverie underBreverieB	reventureBrevenge revisitedBrevenge proBrevenge directorsBrevenants editionBrevenant wingsB	revelatorBrevelations collectionBrev 2BrevBreunionBreturns thisB
returns 3dBreturnalB
return oneBreturn fromB
retrovirusB
retrogradeBretro helixB
retro gameBretro brawlerBretro atariBretriever newB	retrieverBresurreccionBrestricted areaB
restrictedBrestless dreamsBrestlessBresolutiionBresogun heroesBresogun defendersBresistance retributionBresistance fallBresistance enhancedBresistance burningBresistance 3Bresistance 2B	resist orBresistB	reshelledB	reshef ofBreshefBrescue missionBrequiem symphonyB	request 2Brepublic theBrepublic iiBrenowned explorersBrenownedBrengoku theB
rengoku iiBrenewalB	renegadesBrenegade squadronBrenegade mechBremote assaultBremoteBremix hyperBremix 2Bremilore lostBremiloreB
rememoriedBremastered trilogyBreloaded editionB	relics ofBrelicsBrekoilBreincarnation tenseiB
reigns herBreigning evilBreigningBreign 2Bregular showBregularBregicideB	refuelledBreforgedB
reflexionsBreflex editionBreflections chapterBreflectionsB
reflectionB
refine theBrefineBreelected gatBreefsBreefer madnessBreeferB
redux darkBredshirtBredshiftBredout 2016Bredneck racingBredimensionedB	red wingsBred tideBred theB
red shadowB	red sabreBred rockB
red ridingB
red rescueBred lieBred lanternB	red jokerBred hoodB
red dragonBred dogB
red devilsB
red cliffsBred catBred advanceBrecruitBrecords theBrecordsBrecord breakerBreconstruction initiativeBreconstructionBrecon shadowBrecon predatorBrecon phantomsBrecon jungleBrecon desertBrecoilBrecodedBreckoning waywardBreckoning theBreckoning redeemerBreckoning 2Breckless disregardBrecklessBrecettear anB	recettearBrecallBrebuiltBreboundBrebooted theB
rebirth3 vBrebirth3Brebirth2 sistersBrebirth2Brebirth1Brebelstar tacticalB	rebelstarBrebels prisonB	rebels ofBrebel strikeB
rebel copsBrebel collectionBreasonable doomB
reasonableBreasonB	reaper ofBrealpolitiksBrealms winterBrealms vampireB	realm iiiB	really beBreallyBrealizationBreality fightersB
reality 20B
real steelBreal soccerB	real poolB	real mystBreal heroesBreal drivingBreal boxingBreactorBreach remasteredBre mindB	re hollowBre chainBrc carsBrazor freestyleBrazorBrazing stormBrazingB
razes hellBrazesBraystorm hdBraysBrayman hoodlumsB	rayman dsBrayman advanceB	rayman 3dBraycrisis seriesB	raycrisisB
ray gigantB
ray bibbiaBraw dataB
raw dangerBraw 2Braving deadB
ravens cryBraven squadBraven shieldBrave masterBrave horrorBravagedBrats vsB	rats 1968Bratchet deadlockedBrat deadBraskullsBrash jailbreakBrashBrasaBrare replayBrareBrapper remasteredBrapper 2Brapala tournamentB
rapala proB	ransom ofB	ransom exBranko tsukigimesBrankoBrangers superBrangers megaforceBrangers megaBrangers battleB	rangers 2B	ranger vsBranger shadowsBranger guardianBrandom encounterBrandals mondayBrandalsBrandallBrancher evoBrancher advanceB	rancher 4B	rancher 3BranchBrampage puzzleB
rampage dxBrampBrallyxBrally trophyBrally racingBrally onlineBrally expansionB
rally coteBrally challengeB
rally 2012Brally 04BrakuenBraising hellBraisingB	rainsweptBrainfallBrainbow skiesBrainbow curseBrain onBrain ofB	rain moveBrain chroniclesBrain beyondBrain alteredBrain 2Brailworks 3B	railworksBrails acrossB	railroadsBrailroad simulatorBrail simulatorBrailBraider trilogyB
raider iiiBraider babaB	raider 20B
raiden iiiBraiden fightersB	raid thisBrah66 comancheBrah66Bragnarok dsBragland 4x4BraglandBrage iiB
rafa nadalBrafaBraetikonBradiohammerBradio futureBradical rabbitBradical editionBradiation islandB	radiationBradiata storiesBradiataBradiant silvergunBradiant mythologyBradiant dawnBradiance ruinsBradarBrack nBrackBracing xBracing withBracing tourBracing sprintBracing simulatorB
racing neoBracing leagueBracing gearsBracing gameB	racing dsB	racing 3dBracing 2003Bracing 2002Bracers roadB
racer zeroBracer vB
racer typeBracer revengeBracer noBracer dsBracer advanceBracer 7Bracer 64Bracer 6Bracer 4Bracer 3dB
racer 2006B
racer 2004B
racer 2003B
racer 2001Bracer 2Brace proBrace onB	race blueBrace 64Brace 07B	raccoonusBrabiribiBrabbitsBrabbit stewB
rabbids tvBrabbids landBrabbids invasionBrabbids aliveBraams shadowBraamsBr4 ridgeBr4Br2Br1Br onlineBqvBquiz tvB	quiz timeBquintetBquinns revengeBquinnsBquiet weekendB	quiet manB	quickspotB
quest warsB	quest viiBquest viBquest vB	quest theBquest swordsBquest ragnarokBquest ofBquest mysteriousBquest magicalBquest ixBquest ivBquest immortalB	quest iiiB
quest handBquest grimmsBquest 3Bquell mementoBquellBqueen ofBqueen blackB	queen andBquartz zoneBquartzBquartet knightsBquartetBquarterBquarrelBquarantine circularB
quarantineBquantum redshiftBquake remasteredBquake arenaBquadrilateral cowboyBquadrilateralB
qoga knellBqogaBqixBqinBqball billiardsBqballBq2 newBq2Bq shadowBqBpyramidsBpuzzling pagesBpuzzling adventureBpuzzlepaloozaBpuzzle worldBpuzzle scapeBpuzzle piratesB	puzzle inBpuzzle guzzleBpuzzle galaxyBpuzzle attackBpuzzle arcadeBpuyo championsBpuyo 2Bputty squadBputtyBpussBpushmo worldBpush theBpush meBpursuitsBpurrfect dateBpurrfectBpurple riptosBpurple hazeBpuritas cordisBpuritasBpurgeB	purgatoryB	pure rideBpure pinballB	pure holdB
pure chessBpunkBpunishment starBpunisher noBpunisher 2005BpunchoutBpuncherB
punch timeB
punch lineB
punch kingB
punch clubBpunch 2Bpumped primedBpumpedBpulse racerBpull youBpullBpuffy amiyumiBpuffyBpuchi virusBpuchi puchiBpucelle tacticsBpucelleB	pub gamesBpubBpto ivBptoBpsychotoxicBpsychonauts inBpsycho kriegBpsychic spectersBpsychicBps vitaB	ps mobileBpryzm chapterBpryzmBproximity flightB	proximityB	prototheaBprotonovus assaultB
protonovusBprotomanBprophecy remasteredBpromptoBpromise unforgottenBpromise revisitedBprom xxlBprom 2Bproject zeroBproject wingmanBproject titanBproject sylpheedBproject sparkBproject nomadsBproject nimbusBproject miraiBproject justiceBproject igiBproject highriseBproject freedomBproject episodesB	project 1Bprogram managerBprogram makingBprogram enhancedBprofile lennethBprofile covenantB	profile 2Bprofessional racingBprofessional driftBproducing perfectionB	producingBprocyonBprocession toB
processionBprobably stoleBprobablyBproamB
pro seriesB	pro rallyBpro conceptBpro castBpro bowlingB	pro beachBprix seriesBprix challengeBprix 4Bprix 3Bprivateers bountyB
privateersBprison escapeBprism lightBprismBpripyatBprinny presentsB
prinny canBprinny 2B	prinny 12Bprinciple roadBprinciple deluxeBprincess peachBprincess ofBprincess hdBprincess guideBprincess fistfulBprincess exBprincess adventuresBprinces editionB
prince theB	prince iiBprince ducklingB	primus ivBprimusBprimordialsB	primordiaBprimedBprime worldBprime trilogyB	prime theBprime pinballBprime huntersBprime federationBprime editionBprime 3Bprime 2Bprimal huntBprimal furyBpride ofBpride fcBprey theB	prey 2006Bpresents theBpresents nisBpresents lotusBpresents karaokeBpresents fastBpresents dragonsBpresents badB
prepare toBprepareBpremonition originsBpremonition 2B
prelude toBprehistorik manBprehistorikBprehistoric punkBprehistoric movesBpredator requiemBprecursor legacyB	precursorBpraetorians hdBpq2 practicalBpq2Bpq practicalBpqBpp producingBppBpowerup heroesBpowerup foreverBpowerstar golfB	powerstarB
powerslideBpowers unitedBpowers matchmakerBpowerpuff girlsB	powerpuffBpowerful shippudenBpowerfulB
powered byB
powerdromeB
power tourBpower spikeBpower soccerBpower respectBpower iiBpower decadesBpower awakensBpourB
potter forBpotatoBpotata fairyBpotataBpostapocalyptic rpgBpostapocalyptic indieBpostal reduxB
postal iiiB	post voidBpost mortemBpossible kimmunicatorB
possible 2Bportrait ofBportraitBporterBportalsBportal stillBportal runnerBportal pinballB
portable 2Bporsche unleashedBporscheBpopup pilgrimsBpopupBpopulous dsBpopulousBpopolocrois aBpopeye rushBpopeyeBpop volB
pop islandB
pool partyB
pool panicBpool ofBpool 3B	pool 2004Bpoohs rumblyBpoohsB
poodle newBpoodleBpoochy yoshisBpoochyBpony islandBpony friendsB
pong questBpong proBponchosBponchoBpompolic warsBpompolicBpolybiusBpoly bridgeBpolyBpollenBpolice departmentBpolice chasesBpolariumBpolaris sectorB	polaris aB
poker withBpoker tournamentBpoker smashBpoker deluxeB
poker 2008Bpokepark wiiB
pokepark 2B	pokemon yB
pokemon xdB	pokemon xBpokemon uniteBpokemon trozeiBpokemon superBpokemon sunBpokemon stadiumBpokemon soulsilverBpokemon shuffleBpokemon shieldBpokemon rubyBpokemon ranchBpokemon questBpokemon puzzleBpokemon platinumBpokemon pinballBpokemon picrossBpokemon pearlBpokemon omegaBpokemon moonBpokemon leafgreenBpokemon heartgoldBpokemon fireredBpokemon emeraldBpokemon dreamBpokemon diamondBpokemon dashBpokemon conquestBpokemon colosseumBpokemon channelBpokemon cafeBpokemon artBpokemon alphaB
pokedex 3dBpokedexBpoints bulletinBpointsB
point roadBpoint blankBpoint behemothBpoi explorerBpoiBpogo islandBpogoBpoe andBpoeBpod speedzoneBpodBpocketbike racerB
pocketbikeBpocket rumbleBpocket racersBpocket poolBpocket planetBpocket paradiseBpocket footballBpocket editionBpocket colorBpocket cardBpn 03BpnBplus odeBplus mysteriesBplus maidensBplus cutB
plus alphaBplunderBplumeBplazaBplaystation vrBplaystation 3BplayroomB	play4freeBplay theBplay motionBplay heroesB
play chessBplay 2BplatypusBplatoon theBplatinum versionBplatforminesBplatform jumperBplatformBplasticB	plant 530BplantBplanetside coreB
planetbaseBplanetary annihilationB	planetaryB
planet zooBplanet spaceBplanet robobotBplanet puzzleBplanet monstersBplanet minigolfBplanet editionBplanet crashersBplaneswalkers 2015B	planes ofBplanar conquestBplanarBplain sightBplainB	plague ofB
place likeBpiyotamaBpixelsBpixeljunk sidescrollerBpixeljunk raidersBpixeljunk racersBpixeljunk 4amBpixel rippedBpixel heroesB
pixel gearBpixar adventureBpixarB
pitch 2003BpitchB
pirates vsBpirates outlawsBpirates noblesBpirates duelsB	pirates 2Bpirate hunterBpirate coveBpirate actionB
pipe maniaBpipeBpioneers ofBpinobee wingsBpinobeeBpinnacle stationBpinnacleB	ping pongB	ping palsBpineapple smashB	pineappleB
pinball vrBpinball vengeanceBpinball theBpinball rubyBpinball rogueBpinball partyB
pinball ofBpinball landBpinball heroesBpinball fx2Bpinball forgottenBpinball fantasiesBpinball balanceBpinball avengersBpinata troubleBpinata pocketBpinata partyBpilotwings resortBpilotwings 64B
pilot teamBpillarBpilgrimsBpikmin 2Bpikachus adventureBpikachusB
pierre theBpierreBpiece romanceBpiece mansionBpiczle linesBpiczleB
pictoimageB	pictobitsB
pictionaryB	picross sB
picross e4B
picross e3B
picross e2B	picross eB
picross dsBpicklockBpiccoloBphysics deluxeBphysicsBphoto safariBphoto finderB
photo dojoBphoning homeBphoningBphoenotopia awakeningBphoenotopiaBphoenix risingBphoenix festaBphineas andBphineasBphelps pushBphelpsB	phase twoB	phase oneBpharmaBpharaohs curseBphantomsBphantom warBphantom urbanBphantom opsB
phantom ofBphantom hourglassBphantom fortressBphantom dustBphantom detectiveBphantom crashBphantom covertBphantasmB	phantasiaB	phantarukBphalanxBpga europeanBpetpet adventuresBpetpetBpet storiesBpet shopB	pet alienBpes 2021Bpersons nineBpersonsB
persona q2B	persona qBpersona dancingBpersistence vrBpersia revelationsBpersia epilogueBperseus mandateBperseusBperry rhodanBperryBpernBperiod dramaBperiodBperimeter emperorsB	perils ofBperilsBperhapsB
perfectionB	perfectedBperfect chronologyBperathiaB
per minuteB
per asperaBpenumbra requiemBpenumbra overtureBpenumbra blackBpenultimate editionBpenultimateBpennypunching princessBpennypunchingBpenguinsB
pengel theBpengelBpenelopeB	pendragonBpencil puzzleBpencilBpeggle nightsBpeggle dualBpeggle deluxeBpeggle 2Bpeas experienceBpeasBpearl versionBpeanuts movieBpeanutsBpeaksBpeach beachB
peach ballBpeace walkerBpeace rankoBpeace forceBpcB
pax romanaBpaxBpaws onBpawsBpattons campaignBpattonsBpattonBpatterson womensB	pattersonBpatrician ivBpatrician iiiBpatrician iiBpato boxBpatoBpaths toBpathologic classicBpathologic 2Bpathfinder wrathBpath vrBpath toB	patapon 3BpastranaBpassive fistBpassiveBpassionBparty uB	party theB
party starB
party saveBparty onBparty megamixBparty islandB
party hitsB
party gameBparty favorsBparty dsBparty cruiseBparty classicsBparty championsB
party bookBparty bloodBparty animalsBparty aloneBparty advanceBparty 9Bparty 8Bparty 7Bparty 6Bparty 5Bparty 4Bparty 3dBparty 30Bparty 3Bparty 10Bparts 1Bpartners inBpartisans 1941B	partisansB	part timeBpart iBpart 4Bpart 3BparsecsBparrBparks adventureBparksBpark tenormansBpark supersweetBpark rollerBpark patrolB	park letsBpark iiiBpark empireB	paraworldBparanormal stuffB
paranormalBparanoia happinessBparanoiaBparanautical activityBparanauticalBparadoxBparadise theBparadise crackedBparadise cityBparadigmBparableBpapers pleaseBpapersBpaperboyB
paperboundBpaper sorcererBpaper monstersB	paper jamBpaper beastBpaper 2Bpaper 1Bpanzers coldBpanzer clawsBpanties doodBpantiesBpangya fantasyBpangyaBpandoras towerBpandorasBpandora firstBpandariaBpanda legendaryB	pale mistBpaleBpaladins championsBpaladinsBpainkiller resurrectionBpainkiller redemptionBpainkiller overdoseBpainkiller battleBpainfulBpain amusementBpagesBpagan onlineBpaganBpadreBpacpixB	pacman vsBpacman megaBpacman mazeBpacman galagaBpacman collectionBpacman allstarsB	pacman 99BpackedBpack volumeBpack 2Bpacific squadronBpacific rimBpacific riftBpacific fightersBpacific assaultBpac nBpacB	oz beyondBozBoxBown theBowltimate editionB	owltimateBovivoB	overworldB	overwhelmBoverlord raisingBoverlord minionsBoverlord darkBoverloadBoverkills theB	overkillsBoverkill extendedBoverdrive andBovercooked specialBoverclocked aBover vietnamBover germanyBover gBover arnhemB
outtriggerBoutrun2Boutrun onlineBoutpost kalokiBoutpostBoutliveBoutlast whistleblowerBoutlast bundleBoutfitBouter spaceB	outer rimBoutdoor retreatBoutdoor gamesBoutcryBoutbuddies dxB
outbuddiesBoutbreak fileBoutatimeB	out thereBout runBout fromBout anniversaryBoureB	our worldBour garbageBotomedius excellentB	otomediusB
otogi mythBotogi 2Bother mBothelloBostfront 4145BostfrontBosmosBoscBosborne houseBosborneBorwells animalBorwellsBorwellBortaBorphen scionBorphenBorion piratesBorion iiB
orion dinoBorion conquerBorion 3Borigins witchBorigins lelianasBorigins collectionBorigins coldB	origins 2Boriginal warB	origin ofBorigami kingBorigamiBoriental empiresBorientalBorient expressBorientBoriathB
orgarhythmBoreshika taintedBoreshikaBorder legaciesBorder justiceBorder iiB
order deadBorder criminalB
order 1886B
orcs elvesBorchestra ostfrontBorchestra 2BorbzBorbitBorbientBorbBoratorio tangramBoratorioBorangebloodB
orange theBoracleBor serveBor gloryBor beBops warshipBops rezurrectionBops plusBops missionB	ops firstBops essentialsBops escalationBops declassifiedBops commanderBops assaultBops annihilationBopposing frontsBopposingBopoonaBopinionBoperations typhoonBoperations escalationBoperation wintersunBoperation vietnamBoperation videogameBoperation tokyoBoperation thunderBoperation spyBoperation resurrectionBoperation phoenixBoperation pantiesBoperation hiddenBoperation exodusBoperation darknessBoperation brokenBoperation blockadeBoperation babelBoperation arrowheadBoperation abyssBopen seasonBopen meBopen forB	open 2002B
oozi earthBooziBooo ascensionBoooB
ooga boogaB
ontamaramaB
online verBonline thirdBonline stormreachBonline specialBonline siegeBonline shadowsBonline shadowlandsB	online rxB	online reBonline racingBonline minesBonline millenniumBonline lostBonline dragonholdBonline chessBonline blitzkriegBonline blackwoodBonline alienB
online ageBonimusha tacticsBonimusha dawnBonimusha bladeB
onimusha 3B
onimusha 2BoneshotBonechanbara z2B	one worldB
one wickedBone uponBone theBone survivalB
one rebornB	one nightBone mustBone moreBone lineBone hitsBone forB
one fingerBone championshipB
one brokenBone aBone 2001Bon yourBon withB	on wheelsB
on soldierBon rustyBon railsB	on primusBon pearlB	on modernBon helicoptersB
on empathyB	on dragonBon doneB
on deponiaBon deceptionBon daBon cataclysmB	omniverseBomg editionBomgBomens ofBomensBomen ofB	omega theBomega strainBomega stoneB
omega rubyBomega quintetB
omega fiveBomega factorBomega collectionBomasseBolympicsBolympia soireeBolympiaBolliolli2 xlBolliolli switchBollBoliverB
olive townBoliveB
olde worldBoldeB
old schoolBold onesBold huntersB	old clockBold cityBokunokaBokhlosBokamidenBokage shadowBokageBokabuBok koBokBoil rushB
oil empireB	oil buildB	ohsir theBoh myBohBogre theBogre letBogre battleBogre attacksBog sagaBogBoffworld tradingBoffworldBoffroad redneckBoffroad driveBoffpeak cityBoffpeakBofficial wtccBofficersBoffice smashBoffenseBoff revivalB
of zombiesBof zoeBof zinBof zillBof zestiriaB
of zerzuraBof zenBof zehirBof ysBof yogB	of xulimaBof xanaB	of wolvesB
of wishingBof whispersBof westgateBof werewolvesB
of waverlyBof warshipsBof warriorsBof warplanesBof wakfuBof voojuBof vonBof violenceBof villainsBof vestroiaBof vermillionB
of veliousBof valhallaBof valentiaBof undrentideB	of ultronBof tyrimBof tronBof tranquilityBof townsB
of tormentBof tomorrowB	of titansBof timesB	of thornsBof testsB	of terrorBof terraBof tasosB	of taipanB	of taikaiB	of swordsBof swingB
of sushidoBof supremacyB	of suguriBof steeltownBof steamworksB	of sportsB	of spiritB	of spellsB	of spadesBof spacetimeBof spaceB
of sorceryB	of soccerBof sirB
of silenceB	of sigmarBof sightB	of shuggyB	of shirinB
of shinobiBof shikigamiBof sherwoodBof setarrifBof setBof seraB
of satinavBof sapphireB	of sanadaBof salvationBof sagaBof ryzomBof runeterraBof routeBof rotatingBof roseB
of roastedBof ringoBof revelationBof reflectionsB	of reasonBof rayB	of ravensBof raetikonBof qinBof purgatoryB
of pripyatB	of princeB
of plasticB
of piratesBof pigsBof phantasiaBof pernBof perathiaB
of pegasusBof patriotsB
of passionBof pandariaBof ozB
of osborneB	of oriathBof operationsBof operationBof oogaBof oliveBof oldeBof oblivionB
of nowhereBof normandieBof norBof nightmaresB
of newerthBof naxxramasBof naheulbeukBof mythBof mrBof moriaB
of mirrorsBof mirkwoodBof metafalicaB	of martinBof marsBof mapleBof mannyB
of malachiBof magnaBof lyricB	of luclinB	of lovingBof loveB	of londonBof logosB	of lodossBof lodisBof loathBof linkBof liesBof liberationBof legendiaB	of legacyBof lawB
of lagaardBof kusakariBof kumaBof kriBof kingB	of kharakBof keflingsB
of keepersBof kapuBof kaiBof kageBof jesusBof jesseB
of istariaBof innsmouthBof innocenceBof infinityBof industryBof incarnatesBof iceB	of hyruleBof hornsBof heraclesB	of hendonB	of heartsBof heartBof happinessBof hammerwatchB	of hakkonBof guybrushBof gunBof grayskullB
of gravityB	of gracesBof goldBof godBof glassBof gilgamechB	of giantsBof gayBof footballB	of flamesBof flameBof fightingB
of faydwerBof fatesBof exigoBof excaliburBof evidenceBof everquestB	of europeB
of etheriaB	of elemiaBof elementalBof eisenwaldB	of eirudyBof ecclesiaB
of dwarvesBof dungeoneeringB
of dungeonB	of druagaB
of dredmorB
of dragoonBof dragonspearB
of draenorB
of draculaBof doverB	of doctorBof distrustBof dissonanceBof disguiseB
of discordBof dirtBof digitollBof devastationBof detentionBof defianceBof decadenceBof dawnB	of darwinBof daBof crimeB	of corsusBof corruptionBof cooperationB	of cookieB
of contactBof conquestB	of coffinBof clubsBof chernobylB	of changeBof cerberusBof catastropheB
of carrotsBof calamityBof cakeB
of bushidoBof bumboBof bugsB	of brokenBof broadcastBof braveB	of boundsBof boneBof birthBof birdBof bhaalB	of battleB
of aureliaB	of asgardBof armorBof argusB	of arcanaBof arcadiasBof arcB	of arannaBof arB
of anteriaBof annihilatedB	of angmarB	of angelsB
of anarchyBof amnBof ammoB
of americaB
of alrevisBof alonBof almiaB
of alchemyBof ahtBof agonyBof aggressionBof aetheriaBof adventureBof adamB
of abrahamBof abandonedB	odyssey vBodyssey untoldBodyssey nexusB
odyssey ivBodyssey iiiB
odyssey iiBodyssey extendedBodyssey aceB	odyssey 2BodstBoddityBodd odysseyBoddBodamaBoctopus characterBoctopusB	octomaniaBocto expansionBoctoBoctahedron transfixedBoceans heartBoceansBoceanhorn 2B
ocean tillBocean secondBocean integrityB
ocean blueB	ocean bigBocean 4BobsidianB
obscuritasBobscuraB	obra dinnBobraBoblivion afterlifeBoblitusBobiwanBobelix kickBobeliskB	obductionBoasisBnyxquest kindredBnyxquestBnycBny theBny laBnxt lvlBnxtB
nuts boltsB
number oneB
nukem landBnukem criticalBnukem advanceBnukem 64BnucleusBnuclear dawnBntrancedBntBnstrike eliteBnovinewsBnovelistB
novastrikeB	novadromeBnova covertBnovaB	not sleepBnot passBnot feedBnot fallBnosurge plusBnosurge odeBnostradamus theBnostradamusB	nostalgiaBnosferatu theB	nosferatuB	northlandBnorthern strikeBnorthernBnorth konungBnorth enhancedB	norn9 varBnorn9Bnormandy 44B	normandieBnorBnonary gamesBnonaryBnomadsB	nomad theBnomadBnolderB
noitu loveBnoituBnoitaBnoirs hollywoodBnoirsBnoire reeferBnoire nicholsonBnoire goddessB	noby nobyBnoby boyBnoblesB
no shelterB	no longerB	no heroesB
no gravityBno exitB	no escapeBnitroplus blasterzB	nitroplusB	nitrobikeBnis classicsBnisBnirvanaBnippon marathonBnipponBnioh dragonBnioh defiantBnioh completeBnioh collectionBnioh bloodshedsBnintendogs labBnintendogs dalmatianBnintendogs dachshundBnintendogs chihuahuaBnintendo pocketBnintendo landBninjin clashBninjinB	ninjatownBninjas dodgeballBninjas adventuresB
ninjamuraiBninjalaBninja usagimaruBninja starringBninja shodownBninja returnsBninja proamBninja impactBninja fiveoBninja assaultBninja 3Bnine theBnine personsB
nine livesB
nine hoursB
nine doorsBnine breakerBnin2jumpBnimbus completeBnimbusBnikopol secretsBnikopolBnikolis pencilBnikolisBnike kinectBnikeBnightskyB
nightshadeBnights withBnights shadowsBnights journeyBnights hordesBnightmares theBnightmares fromBnightmares completeBnightmare troubadourBnightmare princessBnightmare ofBnightmare inBnightmare editionBnightmare childB	nightlifeB	nightfallBnightcryBnightcrawlerBnightcaster iiBnight yahtzeeBnight watchB
night twinBnight terrorBnight standBnight scrabbleBnight rondoBnight expansionBnight connectBnight beyondBnight battleshipBnight aloneBnight 6Bnight 5Bnigh completeBnicktoons battleBnickelodeon dannyBnicholson electroplatingB	nicholsonB
nibiru ageBnibiruBnhl slapshotB
nhl rivalsBnhl 2k2Bnhl 2k11Bnhl 21Bnfl 2k1B	nexus theBnexuizBnext penelopeB
next orderB	next lifeBnext challengeBnext bigBnexomon extinctionBnexomonBnexagon deathmatchBnexagonB
news flashBnewsBnew zealandB
new worldsB
new visionBnew travelerBnew signB
new rallyxB	new powerBnew pokemonBnew nintendoB
new littleBnew leafB
new jaggedB
new islandBnew internationalBnew horizonsBnew guysB
new gundamB
new grooveBnew generationBnew eyesB	new droidBnew despairB
new cinemaB	new breedB	new bloodBnew beelzebubBnevesB	neversongBneveroutB	nevermindBneverland cardB	neverlandBneverend 2006BneverendBnever yieldB
never stopBnever partyB
never loseB
never giveBnever fightBnever endingBneurovoiderBnetwork transmissionBnetwork punchBnetwork cookB	network 3B	network 2BnetherBnessBnervous brickdownBnervousBnerveB
nerd savesBnerd iBnerd adventuresBneptunia viirBneptunia viiBneptunia victoryB
neptunia uBneptunia reverseBneptunia rebirth3Bneptunia rebirth2Bneptunia rebirth1Bneptunia ppBneptunia mk2B
neptunia 4BneowaveBneoverse trinityBneoverseBneopets theBneopets petpetBneon structBneogeo shockBneogeo metalBneogeo battleBneocronBneo scavengerBneo geoB
neo contraB
neo climaxBnemesis strikeB
nemesis ofBneighborville completeBneighbors fromB	neighborsBnectarisBnecrovisionBnecrodancer nintendoBnecrodancer featuringBnecrobarista finalBnebulaBnebuchadnezzarBncisBnba startingBnba courtsideB
nba ballerBnba 2k1Bnba 10B	naxxramasB	naval warBnaval assaultBnauticalBnatural selectionBnatural disastersBnatsuki chroniclesBnatsukiBnatrolisBnations thronesBnations theBnations riseBnational hockeyBnationalBnation roadBnation dividedBnation apocalypseBnathan drakeBnathanBnat geoBnatBnasiras revengeBnasirasBnascar simracingBnascar kartB	nascar 14B	naruto vsB
naruto theBnaruto riseBnaruto powerfulBnarco terrorBnarcoBnaraka bladepointBnarakaBnapoleons lastBnapoleons greatestBnapoleonic warsB
napoleonicBnapoleon totalB	nantucketBnanostray 2BnanobreakerB	named tomBnamed schoolBnamed fightB
name steamBnameBnakatomi plazaBnakatomiB
naissanceeBnairi towerBnairiBnaheulbeuk theB
naheulbeukBnadal tennisBnadalBn3ii ninetynineBn3iiBn turfB	n surviveBn slashBn ruinBn loadBn ghostsBn funBn blastBmytran warsBmytranBmyths ofBmythsBmythology theBmythology extendedBmythologiesBmyth theBmyth ofBmyth iiiBmyth drannorBmystical ninjaBmysticalBmystic valleyBmystery theBmystery episodeBmystery curseBmysterious timesBmysterious karakuriBmysterious islandBmysterious codexBmysterious citiesBmysterious bookBmysterios menaceB	mysteriosBmysteriaBmystereBmyst vBmyst ivBmysims agentsBmyreBmy worldBmy wayBmy uncleB	my streetB
my spanishBmy radioBmy princessB
my pokemonBmy petBmy horseBmy heartBmy godheadsBmy gameB
my froggerB
my brotherBmxriderB	mxgp3 theBmxgp3B	mxgp 2020Bmvp 07B
mutropolisBmutoBmutant footballBmutant blobsBmusynxBmustang theBmustangBmust answerBmusketeers oneB
musketeersBmusical singBmusical adventureBmusic vrBmusic mixerB	music jamBmusic evolvedBmushroom warsBmushroom 11BmushihimesamaBmuseum virtualBmuseum remixBmuseum megamixBmuseum essentialsB	museum dsB	muse dashBmuseB
muscle theBmuscle marchBmuscle legendsBmusashi samuraiBmurphy adventureBmurphyBmurderous pursuitsB	murderousB	murder ofB	murder inB
murder fbiBmurder clubBmurder cardsBmurasaki babyBmurasakiBmuramasa theBmuramasa rebirthBmurakumo renegadeBmurakumoBmuppets partyB
muppets onBmuppets movieBmunroBmuninB
munchablesBmummy returnsBmultiwinia survivalB
multiwiniaBmultiplayer mapBmultiplayerBmugstersBmugen soulsBmugenBmudrunners aB
mudrunnersBmudds superBmudds deluxeBmudds collectionBmucha luchaBmuchaBms splosionBms sagaBmr xBmr pantsBmr chardishBmoxxis underdomeBmoxxis heistBmowing simulatorBmowingB
mower kidsBmowerBmovies stuntsB	movies 3dBmovie snoopysBmovie nightBmovie adventuresBmoves deadmundsBmove heroesBmove fitnessBmove apeBmouthBmountains ofBmountains downhillB	motorwaysBmotorstorm rcBmotorstorm pacificBmotorstorm apocalypseBmotorsport managerBmotorsport 6Bmotorsport 5Bmotorsport 4Bmotorsport 3Bmotorsport 2Bmotor trendBmotor racingBmotor mayhemB
motor cityB	motoherozBmotogp 4B	motogp 18B	motogp 17B	motogp 14Bmotogp 1011B	motogp 06Bmotocross mcgrathBmotocross maniacsBmotocross madnessBmotocross 2001Bmoto racingBmotivesBmotionsports adrenalineBmotion 2BmothsB
motherloadBmother russiaBmotherBmossBmosquitoBmoseley madBmoseleyBmortemBmorrowind gameBmorphiteBmorphin powerBmorphinBmorphies lawBmorphiesBmorningstar descentBmorningstarBmorning rpgBmorningBmormos curseBmormosBmoriaBmori 2Bmorgana dreamsBmorganaB
morgan andBmorganBmore workoutsBmore trainingB	more jumpBmore friendsBmore editionB
mordor theBmordhauBmordecai andBmordecaiBmorbus gravisBmorbusB
morbid theBmorbidBmoosemanBmoonlight witchBmoonbase commanderBmoonbaseB	moon treeBmoon theBmoon skytreeB	moon saveBmoon projectBmoon oneB	moon moreBmoon huntersBmoon friendsBmoon franticBmoon fortunaBmoon anotherBmoon animalBmoon aBmoon 64B	mooil rigBmooilBmontana musicBmontanaBmontagues mountB	montaguesBmonsters recutBmonsters probablyBmonsters encoreBmonsters deluxeB
monsters aB
monsters 2B
monsterbagBmonster trainBmonster taleBmonster slayerBmonster sealBmonster sanctuaryBmonster monpieceBmonster lairBmonster kingdomBmonster harvestBmonster forceBmonster coliseumBmonster campBmonpieceBmonopoly tycoonBmonopoly partyBmonopoly forBmonophonic menaceB
monophonicBmonkey mayhemBmonkey kingBmonkBmoney powerBmomodora reverieBmomodoraB	moment ofBmomentBmom hidBmomBmole theBmole controlB
mojo reduxBmojo rampageB	mojo jojoB
mogul 2008B
mogul 2003Bmoero crystalBmoeroBmoebius empireBmoebiusBmodified airBmodifiedBmodern worldBmodern timesBmodern militaryBmodern hitsB
modern airB	moco mocoBmoco friendsBmobsB	mobilizedBmobile lightBmobile forcesBmobile armorBmob warsBmob andBmoai betterBmoaiBmma onslaughtBmls extratimeBmlsBmlb 2k13Bmlb 2006Bmlb 2005Bmlb 2004Bmlb 2003Bmlb 15Bmk2BmixerBmitsurugi kamuiB	mitsurugiBmists ofBmistsBmistmareBmister slimeBmister mosquitoBmistB
mission toBmission shockBmission packBmission barbarossaB	mission 4B	mission 3Bmissing sinceBmissing heirB
missing anBmissile furyBmissile crisisBmissile commandB	miss takeBmissBmisfitsBmirrormoon epB
mirrormoonBmirror starringB
mirror iiiB	mirror iiB	mirra bmxBmirkwoodBmirai dxBmiraiBmirage arcaneBmiracle maskBmiracle cureBminors majesticBminorsBminnieBministry ofBministryB
minish capBminishB
miniseriesBminis onBminis marchBminion mayhemBminionBminimumBminimech mayhemBminimechBminiland mayhemBminilandBminigolf touchBminigolf adventuresBmini motorwaysB
mini motorBmini mixB	mini golfBminesweeper flagsBminesweeperBmines ofBminesBminervas denBminervasB
miner warsBminerBminecraft wiiBminecraft newBmine completeB	mindseizeB	mind zeroB	mind pathB	millipedeBmillionheirBmillennium twilightBmillennium girlBmillennium 2001Bmilkmaid ofBmilkmaidBmilitary tacticsBmilitary madnessBmilitantBmiles edgeworthBmighty questBmighty morphinBmighty milkyBmighty flipB
mighty 8thBmidways greatestBmidwaysBmidnight shadowsBmidnight poolBmidnight nowhereBmidnight carnivalBmiddle kingdomBmiddleBmicrosoft trainBmicrosoft golfB	microgameBmickeys speedwayBmickeysB
mickey theB
michonne aBmichael phelpsBmicBmiasmataB
miami viceB	miami lawBmiami collectionBmiaBmevo theBmevoBmetropolis streetB
metropolisBmetronomicon slayBmetroid zeroBmetroid samusBmetroid otherBmetroid fusionBmetroid dreadBmeteos warsBmeteos disneyBmeteor hunterBmeteorBmetalxBmetal xBmetal torrentBmetal smallB
metal sagaBmetal rumbleB	metal maxBmetal geomatrixB
metal fullBmetal fatigueB
metal fakkBmetal dungeonB	metal ageB
metafalicaBmessenger 2001BmesaB
meruru theBmeruru plusBmerchant princeBmerchant marineBmercenaries wingsBmercenaries 3dB	mens roomBmensBmenhir forestB	men worldBmen theyBmen theBmen riseB	men greenBmen atBmen 3Bmen 2Bmemories integralBmemoricks adventuresB	memoricksBmemoria 2013BmemoriaB	memorandaBmemoirBmembraneBmelty bloodBmeltyBmeltdown revolutionBmeltdown remixBmelodiasBmelbits worldBmelbitsBmeiers starshipsBmeiers simgolfBmeiers railroadsBmeiers gettysburgBmeiers alphaB
meiers aceBmegaton rainfallBmegatagmension blancBmegatagmensionBmegaquariumBmegamind ultimateBmegamindBmegabyte punchBmegabyteBmega tunnelB	mega quizB
mega partyBmega neoBmega microgameB	mega coinB
mega brainBmega battleBmeets felicityBmeetsB
meet againBmedievil resurrectionBmedieval piratesBmedieval movesBmedieval lordsBmedieval dynastyBmedic specialBmedicBmedes islandsBmedesBmechwarrior onlineBmechwarrior 5Bmechstermination forceBmechsterminationBmechcommander 2BmechcommanderBmechassault phantomBmechassault 2Bmechanized combatB
mechanizedBmechanic masterBmechanicBmechaBmech pursuitBmech platoonBmeatballBme theBme pullB
me episodeBmdk2Bmdk 2BmdkBmcmorris infiniteBmcmorrisB
mcgrath vsBmcgees aliceBmcgeesBmcgee presentsBmcgeeBmcfarlanes evilB
mcfarlanesB	mc groovzBmcBmazes ofBmazesBmaze madnessBmaze detectiveBmayhem vehicularB
mayhem theBmayan deathBmayan adventureB
maxs brainBmaxsBmaximus chariotBmaximusBmaximum velocityBmaximum remixBmaximum poolBmaximum chaseBmaximum capacityB	maximo vsBmaximo ghostsBmaxiboost onB	maxiboostBmax xenoB
max seasonB
max beyondBmawBmaverick hunterBmaverickB
matterfallBmatrix onlineBmathB
matchmakerBmatch ofBmatadorBmasters sempaiBmasters kaijudoB
masterplanBmaster xBmaster trialsB
master spyBmaster quizBmaster questBmaster partyBmaster overdriveBmaster egyptBmassiraBmass attackBmasquerade shadowsBmasquerade redemptionBmasquerade coteriesBmasquerade bloodlinesBmasquerada songsB
masqueradaB
masochisiaBmasked queenBmaskedBmask 3dBmascaritas ofB
mascaritasBmarzBmarvels ironBmarvellous missB
marvellousBmarvel powersBmartin mystereBmartinBmartiansBmars warBmars matrixBmarlow briggsBmarlowBmarker goldB
marked forBmarkedBmark mcmorrisBmario worldBmario superstarBmario superBmario sunshineBmario stickerBmario puzzleBmario pinballBmario onlineBmario odysseyB	mario mixBmario miracleBmario hoopsBmario expressBmario colorB	mario andBmario allstarsBmarine parkBmarine maniaBmarine fishingBmarching fireBmarchingBmarch toBmarch againBmarble sagaBmarble maniaB	marble itBmarble blastBmarathon durandalBmaple creekBmapleBmap packBmapBmany robotsBmany meB
manuscriptBmanticore galaxyB	manticoreBmansion darkB	mansion 3B	mans landBmanor puzzlingBmanor chaoticBmanny riveraBmannyBmannersBmanic monkeyBmanicBmaniaxBmaniacs advanceBmaniacsBmania 3dBmania 3Bmaneater truthB
mandate ofBmanchester unitedB
manchesterBmanassasBmanager touchBmanager liveBmanager handheldBmanager classicB	manager 3Bmanager 2020Bmanager 2018Bmanager 2016Bmanager 2015Bmanager 2014Bmanager 2013Bmanager 2012Bmanager 2011Bmanager 2010Bmanager 2009Bmanager 2006B
manager 13B
manager 12B
manager 11B
manager 10B
manager 09Bman x8Bman x7Bman x6Bman x5Bman vrBman remasteredBman poweredBman networkBman maverickB	man livesBman legendsB
man escapeBman bassB
man 2econdBmamorukun curseB	mamorukunBmamodo furyBmamboB
mama worldB
mama sweetBmama cookstarB	mama cookBmama 5Bmama 3Bmall tycoonBmalicious fallenBmalgrave incidentBmalgraveBmalachiBmakin magicBmakinBmaker iiBmaker huntingB	maker forB	maker fesBmaker 3Bmake warBmakatuBmakai kingdomBmakaiBmajor minorsBmajestys spiffingBmajestysBmajestic marchBmajestic chessB	mail moleBmailB
maidens ofBmaidensB	maiden ofBmaiden heavenBmahjong talesBmahjong cub3dBmagusBmagrunner darkB	magrunnerBmagnificent trufflepigsBmagnificentBmagnets fullyBmagnetsBmagnetica twistBmagna maidenBmagisterBmagicka wizardBmagicka vietnamBmagick obscuraBmagickBmagicians questB	magiciansBmagical starsignBmagical racingBmagical mirrorBmagical dreamsBmagical beatB
magic xyxxBmagic xBmagic steelBmagic questBmagic powerBmagic pengelB
magic orbzBmagic obeliskBmagic mayhemBmagic ixB	magic iiiBmagic elementsBmagic chessBmages initiationBmagBmaestroBmaelstrom 2007B	maelstromBmadworldB
madness ofBmadness nectarisBmadness graveB	madness 3Bmadness 2003Bmadness 2002Bmadness 2001B	madness 2Bmadballs inBmadballsBmadagascar 3Bmad trixB
mad tracksBmad ratB
mad moxxisBmad maestroBmad dashBmacross sagaBmacrossBmacleans mercuryBmacleansBmachtBmachines elementsB
machines 3B
machines 2Bmachineguns arcadeBmachinegunsB
machine vrBmachine rogueBmachine forBmachine 2012Bmachine 2008BmachinariumBmachiavillainBmach modifiedBmachB	macdonaldBmaboshis arcadeBmaboshisB	mable theBmableBmBlyricBlyoko questB
lyoko fallBlydiaBlvl editionBlvlBluxpainB	luxor theBluxor 2B	lust fromBlust forBlupus empireBlupusB	lupin theBlupinBlunateas veilBlunateasBlunar legendBlunar knightsBlunar dragonBlunar 2Bluna theBlunaBluminous avengerBlumino cityBluminoBlumines supernovaBlumines plusBlumines liveB
lumines iiBlumines electronicBluminaries ofB
luminariesBlumeBlullaby episodeBlullabyBluigi uBluigi partnersBluigi paperBluigi dreamBluftwaffe squadronB	luftwaffeB	lufia theBlufia curseBlucreBluclinBluckslingerB	lucius iiBlucha mascaritasBlucha libreB
lucha furyBlowriderBlow roadBlowBlovingBlovely planetBlovelyBlove maxBlove katamariB	love golfBlove devolutionBlove andBlotus challengeBlostwinds winterBlost vikingsBlost valleyBlost swordsB	lost soulB	lost songB
lost ruinsB
lost realmBlost paradiseB
lost orbitBlost onBlost odysseyB
lost oceanB
lost magicBlost legacyBlost kingdomBlost horizonB	lost hoboBlost grimoiresB	lost godsB	lost girlBlost dutchmansBlost dungeonsB
lost crownBlost colonyB	lost codeBlost citiesBlost chroniclesBlost cavernBlost bordersB
lost atlasBlost ageBlosersBloser ultimateBloserB	lose hopeBloseBlords enhancedBlords buildBlord 2Bloopy landscapesBloopyB	loop heroBloopB	loons theBloonsBlooking throughBlookingBlongest roadBlongest nightBlongest fiveBlongest dayBlonger homeBlongerB
long nightBlong journeyB	long goneBlonely mountainsBlonelyB	lone echoBlondon detectiveB
lomu rugbyBlomuB	lol neverBlolBlokiB	logy plusBlogy alchemistsBlogsBlogosBlogans shadowBlogansB
lodoss warBlodossBlodisBlocoroco midnightBlocoroco cocorecchoB
locomotionB	lococycleBlocks questBlocksBlock onB	loch nessBlochBloath nolderBloathBloading humanBloadingBloadBlivin largeBlivinBlive reloadedB	live homeB	live fallBlive battlefestBlive arcadeB
littlewoodBlittlebigplanet psBlittlebigplanet kartingBlittlebigplanet 3Blittlebigplanet 2Blittle witchBlittle townBlittle leagueBlittle friendsB
little bigBlittle battlersBlittle acornsBlitBlisa theBlisaB
lips partyBlips numberBlionheart kingsB	lionheartBlinks crossbowBlinks awakeningB
links 2004B
links 2003B
links 2001Blink toBlink betweenBliningBlines dxB	linelightB
lineage ofB
lineage iiBline defenderBline coloringBline armoredBlincoln mustBlincolnB	lilt lineBliltBlilly lookingBlillyBlike sonB	like homeBlike fatherBlightyear toB
lightspeedBlightsaber duelsB
lightsaberBlightning bolt
??	
Const_5Const*
_output_shapes

:??*
dtype0	*??	
value??	B??		??"??	                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	       
      
      
      
      
      
      
      
      
      	
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
       
      !
      "
      #
      $
      %
      &
      '
      (
      )
      *
      +
      ,
      -
      .
      /
      0
      1
      2
      3
      4
      5
      6
      7
      8
      9
      :
      ;
      <
      =
      >
      ?
      @
      A
      B
      C
      D
      E
      F
      G
      H
      I
      J
      K
      L
      M
      N
      O
      P
      Q
      R
      S
      T
      U
      V
      W
      X
      Y
      Z
      [
      \
      ]
      ^
      _
      `
      a
      b
      c
      d
      e
      f
      g
      h
      i
      j
      k
      l
      m
      n
      o
      p
      q
      r
      s
      t
      u
      v
      w
      x
      y
      z
      {
      |
      }
      ~
      
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                                      	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?        !      !      !      !      !      !      !      !      !      	!      
!      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !       !      !!      "!      #!      $!      %!      &!      '!      (!      )!      *!      +!      ,!      -!      .!      /!      0!      1!      2!      3!      4!      5!      6!      7!      8!      9!      :!      ;!      <!      =!      >!      ?!      @!      A!      B!      C!      D!      E!      F!      G!      H!      I!      J!      K!      L!      M!      N!      O!      P!      Q!      R!      S!      T!      U!      V!      W!      X!      Y!      Z!      [!      \!      ]!      ^!      _!      `!      a!      b!      c!      d!      e!      f!      g!      h!      i!      j!      k!      l!      m!      n!      o!      p!      q!      r!      s!      t!      u!      v!      w!      x!      y!      z!      {!      |!      }!      ~!      !      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!       "      "      "      "      "      "      "      "      "      	"      
"      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "       "      !"      ""      #"      $"      %"      &"      '"      ("      )"      *"      +"      ,"      -"      ."      /"      0"      1"      2"      3"      4"      5"      6"      7"      8"      9"      :"      ;"      <"      ="      >"      ?"      @"      A"      B"      C"      D"      E"      F"      G"      H"      I"      J"      K"      L"      M"      N"      O"      P"      Q"      R"      S"      T"      U"      V"      W"      X"      Y"      Z"      ["      \"      ]"      ^"      _"      `"      a"      b"      c"      d"      e"      f"      g"      h"      i"      j"      k"      l"      m"      n"      o"      p"      q"      r"      s"      t"      u"      v"      w"      x"      y"      z"      {"      |"      }"      ~"      "      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"       #      #      #      #      #      #      #      #      #      	#      
#      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #       #      !#      "#      ##      $#      %#      &#      '#      (#      )#      *#      +#      ,#      -#      .#      /#      0#      1#      2#      3#      4#      5#      6#      7#      8#      9#      :#      ;#      <#      =#      >#      ?#      @#      A#      B#      C#      D#      E#      F#      G#      H#      I#      J#      K#      L#      M#      N#      O#      P#      Q#      R#      S#      T#      U#      V#      W#      X#      Y#      Z#      [#      \#      ]#      ^#      _#      `#      a#      b#      c#      d#      e#      f#      g#      h#      i#      j#      k#      l#      m#      n#      o#      p#      q#      r#      s#      t#      u#      v#      w#      x#      y#      z#      {#      |#      }#      ~#      #      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#       $      $      $      $      $      $      $      $      $      	$      
$      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $       $      !$      "$      #$      $$      %$      &$      '$      ($      )$      *$      +$      ,$      -$      .$      /$      0$      1$      2$      3$      4$      5$      6$      7$      8$      9$      :$      ;$      <$      =$      >$      ?$      @$      A$      B$      C$      D$      E$      F$      G$      H$      I$      J$      K$      L$      M$      N$      O$      P$      Q$      R$      S$      T$      U$      V$      W$      X$      Y$      Z$      [$      \$      ]$      ^$      _$      `$      a$      b$      c$      d$      e$      f$      g$      h$      i$      j$      k$      l$      m$      n$      o$      p$      q$      r$      s$      t$      u$      v$      w$      x$      y$      z$      {$      |$      }$      ~$      $      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$       %      %      %      %      %      %      %      %      %      	%      
%      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %       %      !%      "%      #%      $%      %%      &%      '%      (%      )%      *%      +%      ,%      -%      .%      /%      0%      1%      2%      3%      4%      5%      6%      7%      8%      9%      :%      ;%      <%      =%      >%      ?%      @%      A%      B%      C%      D%      E%      F%      G%      H%      I%      J%      K%      L%      M%      N%      O%      P%      Q%      R%      S%      T%      U%      V%      W%      X%      Y%      Z%      [%      \%      ]%      ^%      _%      `%      a%      b%      c%      d%      e%      f%      g%      h%      i%      j%      k%      l%      m%      n%      o%      p%      q%      r%      s%      t%      u%      v%      w%      x%      y%      z%      {%      |%      }%      ~%      %      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%       &      &      &      &      &      &      &      &      &      	&      
&      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &       &      !&      "&      #&      $&      %&      &&      '&      (&      )&      *&      +&      ,&      -&      .&      /&      0&      1&      2&      3&      4&      5&      6&      7&      8&      9&      :&      ;&      <&      =&      >&      ?&      @&      A&      B&      C&      D&      E&      F&      G&      H&      I&      J&      K&      L&      M&      N&      O&      P&      Q&      R&      S&      T&      U&      V&      W&      X&      Y&      Z&      [&      \&      ]&      ^&      _&      `&      a&      b&      c&      d&      e&      f&      g&      h&      i&      j&      k&      l&      m&      n&      o&      p&      q&      r&      s&      t&      u&      v&      w&      x&      y&      z&      {&      |&      }&      ~&      &      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&       '      '      '      '      '      '      '      '      '      	'      
'      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '      '       '      !'      "'      #'      $'      %'      &'      ''      ('      )'      *'      +'      ,'      -'      .'      /'      0'      1'      2'      3'      4'      5'      6'      7'      8'      9'      :'      ;'      <'      ='      >'      ?'      @'      A'      B'      C'      D'      E'      F'      G'      H'      I'      J'      K'      L'      M'      N'      O'      P'      Q'      R'      S'      T'      U'      V'      W'      X'      Y'      Z'      ['      \'      ]'      ^'      _'      `'      a'      b'      c'      d'      e'      f'      g'      h'      i'      j'      k'      l'      m'      n'      o'      p'      q'      r'      s'      t'      u'      v'      w'      x'      y'      z'      {'      |'      }'      ~'      '      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'      ?'       (      (      (      (      (      (      (      (      (      	(      
(      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (      (       (      !(      "(      #(      $(      %(      &(      '(      ((      )(      *(      +(      ,(      -(      .(      /(      0(      1(      2(      3(      4(      5(      6(      7(      8(      9(      :(      ;(      <(      =(      >(      ?(      @(      A(      B(      C(      D(      E(      F(      G(      H(      I(      J(      K(      L(      M(      N(      O(      P(      Q(      R(      S(      T(      U(      V(      W(      X(      Y(      Z(      [(      \(      ](      ^(      _(      `(      a(      b(      c(      d(      e(      f(      g(      h(      i(      j(      k(      l(      m(      n(      o(      p(      q(      r(      s(      t(      u(      v(      w(      x(      y(      z(      {(      |(      }(      ~(      (      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(      ?(       )      )      )      )      )      )      )      )      )      	)      
)      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )      )       )      !)      ")      #)      $)      %)      &)      ')      ()      ))      *)      +)      ,)      -)      .)      /)      0)      1)      2)      3)      4)      5)      6)      7)      8)      9)      :)      ;)      <)      =)      >)      ?)      @)      A)      B)      C)      D)      E)      F)      G)      H)      I)      J)      K)      L)      M)      N)      O)      P)      Q)      R)      S)      T)      U)      V)      W)      X)      Y)      Z)      [)      \)      ])      ^)      _)      `)      a)      b)      c)      d)      e)      f)      g)      h)      i)      j)      k)      l)      m)      n)      o)      p)      q)      r)      s)      t)      u)      v)      w)      x)      y)      z)      {)      |)      })      ~)      )      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)      ?)       *      *      *      *      *      *      *      *      *      	*      
*      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *      *       *      !*      "*      #*      $*      %*      &*      '*      (*      )*      **      +*      ,*      -*      .*      /*      0*      1*      2*      3*      4*      5*      6*      7*      8*      9*      :*      ;*      <*      =*      >*      ?*      @*      A*      B*      C*      D*      E*      F*      G*      H*      I*      J*      K*      L*      M*      N*      O*      P*      Q*      R*      S*      T*      U*      V*      W*      X*      Y*      Z*      [*      \*      ]*      ^*      _*      `*      a*      b*      c*      d*      e*      f*      g*      h*      i*      j*      k*      l*      m*      n*      o*      p*      q*      r*      s*      t*      u*      v*      w*      x*      y*      z*      {*      |*      }*      ~*      *      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*      ?*       +      +      +      +      +      +      +      +      +      	+      
+      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +      +       +      !+      "+      #+      $+      %+      &+      '+      (+      )+      *+      ++      ,+      -+      .+      /+      0+      1+      2+      3+      4+      5+      6+      7+      8+      9+      :+      ;+      <+      =+      >+      ?+      @+      A+      B+      C+      D+      E+      F+      G+      H+      I+      J+      K+      L+      M+      N+      O+      P+      Q+      R+      S+      T+      U+      V+      W+      X+      Y+      Z+      [+      \+      ]+      ^+      _+      `+      a+      b+      c+      d+      e+      f+      g+      h+      i+      j+      k+      l+      m+      n+      o+      p+      q+      r+      s+      t+      u+      v+      w+      x+      y+      z+      {+      |+      }+      ~+      +      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+      ?+       ,      ,      ,      ,      ,      ,      ,      ,      ,      	,      
,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,      ,       ,      !,      ",      #,      $,      %,      &,      ',      (,      ),      *,      +,      ,,      -,      .,      /,      0,      1,      2,      3,      4,      5,      6,      7,      8,      9,      :,      ;,      <,      =,      >,      ?,      @,      A,      B,      C,      D,      E,      F,      G,      H,      I,      J,      K,      L,      M,      N,      O,      P,      Q,      R,      S,      T,      U,      V,      W,      X,      Y,      Z,      [,      \,      ],      ^,      _,      `,      a,      b,      c,      d,      e,      f,      g,      h,      i,      j,      k,      l,      m,      n,      o,      p,      q,      r,      s,      t,      u,      v,      w,      x,      y,      z,      {,      |,      },      ~,      ,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,      ?,       -      -      -      -      -      -      -      -      -      	-      
-      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -      -       -      !-      "-      #-      $-      %-      &-      '-      (-      )-      *-      +-      ,-      --      .-      /-      0-      1-      2-      3-      4-      5-      6-      7-      8-      9-      :-      ;-      <-      =-      >-      ?-      @-      A-      B-      C-      D-      E-      F-      G-      H-      I-      J-      K-      L-      M-      N-      O-      P-      Q-      R-      S-      T-      U-      V-      W-      X-      Y-      Z-      [-      \-      ]-      ^-      _-      `-      a-      b-      c-      d-      e-      f-      g-      h-      i-      j-      k-      l-      m-      n-      o-      p-      q-      r-      s-      t-      u-      v-      w-      x-      y-      z-      {-      |-      }-      ~-      -      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-      ?-       .      .      .      .      .      .      .      .      .      	.      
.      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .      .       .      !.      ".      #.      $.      %.      &.      '.      (.      ).      *.      +.      ,.      -.      ..      /.      0.      1.      2.      3.      4.      5.      6.      7.      8.      9.      :.      ;.      <.      =.      >.      ?.      @.      A.      B.      C.      D.      E.      F.      G.      H.      I.      J.      K.      L.      M.      N.      O.      P.      Q.      R.      S.      T.      U.      V.      W.      X.      Y.      Z.      [.      \.      ].      ^.      _.      `.      a.      b.      c.      d.      e.      f.      g.      h.      i.      j.      k.      l.      m.      n.      o.      p.      q.      r.      s.      t.      u.      v.      w.      x.      y.      z.      {.      |.      }.      ~.      .      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.      ?.       /      /      /      /      /      /      /      /      /      	/      
/      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /      /       /      !/      "/      #/      $/      %/      &/      '/      (/      )/      */      +/      ,/      -/      ./      //      0/      1/      2/      3/      4/      5/      6/      7/      8/      9/      :/      ;/      </      =/      >/      ?/      @/      A/      B/      C/      D/      E/      F/      G/      H/      I/      J/      K/      L/      M/      N/      O/      P/      Q/      R/      S/      T/      U/      V/      W/      X/      Y/      Z/      [/      \/      ]/      ^/      _/      `/      a/      b/      c/      d/      e/      f/      g/      h/      i/      j/      k/      l/      m/      n/      o/      p/      q/      r/      s/      t/      u/      v/      w/      x/      y/      z/      {/      |/      }/      ~/      /      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/      ?/       0      0      0      0      0      0      0      0      0      	0      
0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0      0       0      !0      "0      #0      $0      %0      &0      '0      (0      )0      *0      +0      ,0      -0      .0      /0      00      10      20      30      40      50      60      70      80      90      :0      ;0      <0      =0      >0      ?0      @0      A0      B0      C0      D0      E0      F0      G0      H0      I0      J0      K0      L0      M0      N0      O0      P0      Q0      R0      S0      T0      U0      V0      W0      X0      Y0      Z0      [0      \0      ]0      ^0      _0      `0      a0      b0      c0      d0      e0      f0      g0      h0      i0      j0      k0      l0      m0      n0      o0      p0      q0      r0      s0      t0      u0      v0      w0      x0      y0      z0      {0      |0      }0      ~0      0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0      ?0       1      1      1      1      1      1      1      1      1      	1      
1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1      1       1      !1      "1      #1      $1      %1      &1      '1      (1      )1      *1      +1      ,1      -1      .1      /1      01      11      21      31      41      51      61      71      81      91      :1      ;1      <1      =1      >1      ?1      @1      A1      B1      C1      D1      E1      F1      G1      H1      I1      J1      K1      L1      M1      N1      O1      P1      Q1      R1      S1      T1      U1      V1      W1      X1      Y1      Z1      [1      \1      ]1      ^1      _1      `1      a1      b1      c1      d1      e1      f1      g1      h1      i1      j1      k1      l1      m1      n1      o1      p1      q1      r1      s1      t1      u1      v1      w1      x1      y1      z1      {1      |1      }1      ~1      1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1      ?1       2      2      2      2      2      2      2      2      2      	2      
2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2      2       2      !2      "2      #2      $2      %2      &2      '2      (2      )2      *2      +2      ,2      -2      .2      /2      02      12      22      32      42      52      62      72      82      92      :2      ;2      <2      =2      >2      ?2      @2      A2      B2      C2      D2      E2      F2      G2      H2      I2      J2      K2      L2      M2      N2      O2      P2      Q2      R2      S2      T2      U2      V2      W2      X2      Y2      Z2      [2      \2      ]2      ^2      _2      `2      a2      b2      c2      d2      e2      f2      g2      h2      i2      j2      k2      l2      m2      n2      o2      p2      q2      r2      s2      t2      u2      v2      w2      x2      y2      z2      {2      |2      }2      ~2      2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2      ?2       3      3      3      3      3      3      3      3      3      	3      
3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3      3       3      !3      "3      #3      $3      %3      &3      '3      (3      )3      *3      +3      ,3      -3      .3      /3      03      13      23      33      43      53      63      73      83      93      :3      ;3      <3      =3      >3      ?3      @3      A3      B3      C3      D3      E3      F3      G3      H3      I3      J3      K3      L3      M3      N3      O3      P3      Q3      R3      S3      T3      U3      V3      W3      X3      Y3      Z3      [3      \3      ]3      ^3      _3      `3      a3      b3      c3      d3      e3      f3      g3      h3      i3      j3      k3      l3      m3      n3      o3      p3      q3      r3      s3      t3      u3      v3      w3      x3      y3      z3      {3      |3      }3      ~3      3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3      ?3       4      4      4      4      4      4      4      4      4      	4      
4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4      4       4      !4      "4      #4      $4      %4      &4      '4      (4      )4      *4      +4      ,4      -4      .4      /4      04      14      24      34      44      54      64      74      84      94      :4      ;4      <4      =4      >4      ?4      @4      A4      B4      C4      D4      E4      F4      G4      H4      I4      J4      K4      L4      M4      N4      O4      P4      Q4      R4      S4      T4      U4      V4      W4      X4      Y4      Z4      [4      \4      ]4      ^4      _4      `4      a4      b4      c4      d4      e4      f4      g4      h4      i4      j4      k4      l4      m4      n4      o4      p4      q4      r4      s4      t4      u4      v4      w4      x4      y4      z4      {4      |4      }4      ~4      4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4      ?4       5      5      5      5      5      5      5      5      5      	5      
5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5      5       5      !5      "5      #5      $5      %5      &5      '5      (5      )5      *5      +5      ,5      -5      .5      /5      05      15      25      35      45      55      65      75      85      95      :5      ;5      <5      =5      >5      ?5      @5      A5      B5      C5      D5      E5      F5      G5      H5      I5      J5      K5      L5      M5      N5      O5      P5      Q5      R5      S5      T5      U5      V5      W5      X5      Y5      Z5      [5      \5      ]5      ^5      _5      `5      a5      b5      c5      d5      e5      f5      g5      h5      i5      j5      k5      l5      m5      n5      o5      p5      q5      r5      s5      t5      u5      v5      w5      x5      y5      z5      {5      |5      }5      ~5      5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5      ?5       6      6      6      6      6      6      6      6      6      	6      
6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6      6       6      !6      "6      #6      $6      %6      &6      '6      (6      )6      *6      +6      ,6      -6      .6      /6      06      16      26      36      46      56      66      76      86      96      :6      ;6      <6      =6      >6      ?6      @6      A6      B6      C6      D6      E6      F6      G6      H6      I6      J6      K6      L6      M6      N6      O6      P6      Q6      R6      S6      T6      U6      V6      W6      X6      Y6      Z6      [6      \6      ]6      ^6      _6      `6      a6      b6      c6      d6      e6      f6      g6      h6      i6      j6      k6      l6      m6      n6      o6      p6      q6      r6      s6      t6      u6      v6      w6      x6      y6      z6      {6      |6      }6      ~6      6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6      ?6       7      7      7      7      7      7      7      7      7      	7      
7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7      7       7      !7      "7      #7      $7      %7      &7      '7      (7      )7      *7      +7      ,7      -7      .7      /7      07      17      27      37      47      57      67      77      87      97      :7      ;7      <7      =7      >7      ?7      @7      A7      B7      C7      D7      E7      F7      G7      H7      I7      J7      K7      L7      M7      N7      O7      P7      Q7      R7      S7      T7      U7      V7      W7      X7      Y7      Z7      [7      \7      ]7      ^7      _7      `7      a7      b7      c7      d7      e7      f7      g7      h7      i7      j7      k7      l7      m7      n7      o7      p7      q7      r7      s7      t7      u7      v7      w7      x7      y7      z7      {7      |7      }7      ~7      7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7      ?7       8      8      8      8      8      8      8      8      8      	8      
8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8      8       8      !8      "8      #8      $8      %8      &8      '8      (8      )8      *8      +8      ,8      -8      .8      /8      08      18      28      38      48      58      68      78      88      98      :8      ;8      <8      =8      >8      ?8      @8      A8      B8      C8      D8      E8      F8      G8      H8      I8      J8      K8      L8      M8      N8      O8      P8      Q8      R8      S8      T8      U8      V8      W8      X8      Y8      Z8      [8      \8      ]8      ^8      _8      `8      a8      b8      c8      d8      e8      f8      g8      h8      i8      j8      k8      l8      m8      n8      o8      p8      q8      r8      s8      t8      u8      v8      w8      x8      y8      z8      {8      |8      }8      ~8      8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8      ?8       9      9      9      9      9      9      9      9      9      	9      
9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9      9       9      !9      "9      #9      $9      %9      &9      '9      (9      )9      *9      +9      ,9      -9      .9      /9      09      19      29      39      49      59      69      79      89      99      :9      ;9      <9      =9      >9      ?9      @9      A9      B9      C9      D9      E9      F9      G9      H9      I9      J9      K9      L9      M9      N9      O9      P9      Q9      R9      S9      T9      U9      V9      W9      X9      Y9      Z9      [9      \9      ]9      ^9      _9      `9      a9      b9      c9      d9      e9      f9      g9      h9      i9      j9      k9      l9      m9      n9      o9      p9      q9      r9      s9      t9      u9      v9      w9      x9      y9      z9      {9      |9      }9      ~9      9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9      ?9       :      :      :      :      :      :      :      :      :      	:      
:      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :      :       :      !:      ":      #:      $:      %:      &:      ':      (:      ):      *:      +:      ,:      -:      .:      /:      0:      1:      2:      3:      4:      5:      6:      7:      8:      9:      ::      ;:      <:      =:      >:      ?:      @:      A:      B:      C:      D:      E:      F:      G:      H:      I:      J:      K:      L:      M:      N:      O:      P:      Q:      R:      S:      T:      U:      V:      W:      X:      Y:      Z:      [:      \:      ]:      ^:      _:      `:      a:      b:      c:      d:      e:      f:      g:      h:      i:      j:      k:      l:      m:      n:      o:      p:      q:      r:      s:      t:      u:      v:      w:      x:      y:      z:      {:      |:      }:      ~:      :      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:      ?:       ;      ;      ;      ;      ;      ;      ;      ;      ;      	;      
;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;      ;       ;      !;      ";      #;      $;      %;      &;      ';      (;      );      *;      +;      ,;      -;      .;      /;      0;      1;      2;      3;      4;      5;      6;      7;      8;      9;      :;      ;;      <;      =;      >;      ?;      @;      A;      B;      C;      D;      E;      F;      G;      H;      I;      J;      K;      L;      M;      N;      O;      P;      Q;      R;      S;      T;      U;      V;      W;      X;      Y;      Z;      [;      \;      ];      ^;      _;      `;      a;      b;      c;      d;      e;      f;      g;      h;      i;      j;      k;      l;      m;      n;      o;      p;      q;      r;      s;      t;      u;      v;      w;      x;      y;      z;      {;      |;      };      ~;      ;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;      ?;       <      <      <      <      <      <      <      <      <      	<      
<      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <      <       <      !<      "<      #<      $<      %<      &<      '<      (<      )<      *<      +<      ,<      -<      .<      /<      0<      1<      2<      3<      4<      5<      6<      7<      8<      9<      :<      ;<      <<      =<      ><      ?<      @<      A<      B<      C<      D<      E<      F<      G<      H<      I<      J<      K<      L<      M<      N<      O<      P<      Q<      R<      S<      T<      U<      V<      W<      X<      Y<      Z<      [<      \<      ]<      ^<      _<      `<      a<      b<      c<      d<      e<      f<      g<      h<      i<      j<      k<      l<      m<      n<      o<      p<      q<      r<      s<      t<      u<      v<      w<      x<      y<      z<      {<      |<      }<      ~<      <      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<      ?<       =      =      =      =      =      =      =      =      =      	=      
=      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =      =       =      !=      "=      #=      $=      %=      &=      '=      (=      )=      *=      +=      ,=      -=      .=      /=      0=      1=      2=      3=      4=      5=      6=      7=      8=      9=      :=      ;=      <=      ==      >=      ?=      @=      A=      B=      C=      D=      E=      F=      G=      H=      I=      J=      K=      L=      M=      N=      O=      P=      Q=      R=      S=      T=      U=      V=      W=      X=      Y=      Z=      [=      \=      ]=      ^=      _=      `=      a=      b=      c=      d=      e=      f=      g=      h=      i=      j=      k=      l=      m=      n=      o=      p=      q=      r=      s=      t=      u=      v=      w=      x=      y=      z=      {=      |=      }=      ~=      =      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=      ?=       >      >      >      >      >      >      >      >      >      	>      
>      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >      >       >      !>      ">      #>      $>      %>      &>      '>      (>      )>      *>      +>      ,>      ->      .>      />      0>      1>      2>      3>      4>      5>      6>      7>      8>      9>      :>      ;>      <>      =>      >>      ?>      @>      A>      B>      C>      D>      E>      F>      G>      H>      I>      J>      K>      L>      M>      N>      O>      P>      Q>      R>      S>      T>      U>      V>      W>      X>      Y>      Z>      [>      \>      ]>      ^>      _>      `>      a>      b>      c>      d>      e>      f>      g>      h>      i>      j>      k>      l>      m>      n>      o>      p>      q>      r>      s>      t>      u>      v>      w>      x>      y>      z>      {>      |>      }>      ~>      >      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>      ?>       ?      ?      ?      ?      ?      ?      ?      ?      ?      	?      
?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       ?      !?      "?      #?      $?      %?      &?      '?      (?      )?      *?      +?      ,?      -?      .?      /?      0?      1?      2?      3?      4?      5?      6?      7?      8?      9?      :?      ;?      <?      =?      >?      ??      @?      A?      B?      C?      D?      E?      F?      G?      H?      I?      J?      K?      L?      M?      N?      O?      P?      Q?      R?      S?      T?      U?      V?      W?      X?      Y?      Z?      [?      \?      ]?      ^?      _?      `?      a?      b?      c?      d?      e?      f?      g?      h?      i?      j?      k?      l?      m?      n?      o?      p?      q?      r?      s?      t?      u?      v?      w?      x?      y?      z?      {?      |?      }?      ~?      ?      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??      ??       @      @      @      @      @      @      @      @      @      	@      
@      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @       @      !@      "@      #@      $@      %@      &@      '@      (@      )@      *@      +@      ,@      -@      .@      /@      0@      1@      2@      3@      4@      5@      6@      7@      8@      9@      :@      ;@      <@      =@      >@      ?@      @@      A@      B@      C@      D@      E@      F@      G@      H@      I@      J@      K@      L@      M@      N@      O@      P@      Q@      R@      S@      T@      U@      V@      W@      X@      Y@      Z@      [@      \@      ]@      ^@      _@      `@      a@      b@      c@      d@      e@      f@      g@      h@      i@      j@      k@      l@      m@      n@      o@      p@      q@      r@      s@      t@      u@      v@      w@      x@      y@      z@      {@      |@      }@      ~@      @      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@      ?@       A      A      A      A      A      A      A      A      A      	A      
A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A      A       A      !A      "A      #A      $A      %A      &A      'A      (A      )A      *A      +A      ,A      -A      .A      /A      0A      1A      2A      3A      4A      5A      6A      7A      8A      9A      :A      ;A      <A      =A      >A      ?A      @A      AA      BA      CA      DA      EA      FA      GA      HA      IA      JA      KA      LA      MA      NA      OA      PA      QA      RA      SA      TA      UA      VA      WA      XA      YA      ZA      [A      \A      ]A      ^A      _A      `A      aA      bA      cA      dA      eA      fA      gA      hA      iA      jA      kA      lA      mA      nA      oA      pA      qA      rA      sA      tA      uA      vA      wA      xA      yA      zA      {A      |A      }A      ~A      A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A      ?A       B      B      B      B      B      B      B      B      B      	B      
B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B      B       B      !B      "B      #B      $B      %B      &B      'B      (B      )B      *B      +B      ,B      -B      .B      /B      0B      1B      2B      3B      4B      5B      6B      7B      8B      9B      :B      ;B      <B      =B      >B      ?B      @B      AB      BB      CB      DB      EB      FB      GB      HB      IB      JB      KB      LB      MB      NB      OB      PB      QB      RB      SB      TB      UB      VB      WB      XB      YB      ZB      [B      \B      ]B      ^B      _B      `B      aB      bB      cB      dB      eB      fB      gB      hB      iB      jB      kB      lB      mB      nB      oB      pB      qB      rB      sB      tB      uB      vB      wB      xB      yB      zB      {B      |B      }B      ~B      B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B      ?B       C      C      C      C      C      C      C      C      C      	C      
C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C      C       C      !C      "C      #C      $C      %C      &C      'C      (C      )C      *C      +C      ,C      -C      .C      /C      0C      1C      2C      3C      4C      5C      6C      7C      8C      9C      :C      ;C      <C      =C      >C      ?C      @C      AC      BC      CC      DC      EC      FC      GC      HC      IC      JC      KC      LC      MC      NC      OC      PC      QC      RC      SC      TC      UC      VC      WC      XC      YC      ZC      [C      \C      ]C      ^C      _C      `C      aC      bC      cC      dC      eC      fC      gC      hC      iC      jC      kC      lC      mC      nC      oC      pC      qC      rC      sC      tC      uC      vC      wC      xC      yC      zC      {C      |C      }C      ~C      C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C      ?C       D      D      D      D      D      D      D      D      D      	D      
D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D      D       D      !D      "D      #D      $D      %D      &D      'D      (D      )D      *D      +D      ,D      -D      .D      /D      0D      1D      2D      3D      4D      5D      6D      7D      8D      9D      :D      ;D      <D      =D      >D      ?D      @D      AD      BD      CD      DD      ED      FD      GD      HD      ID      JD      KD      LD      MD      ND      OD      PD      QD      RD      SD      TD      UD      VD      WD      XD      YD      ZD      [D      \D      ]D      ^D      _D      `D      aD      bD      cD      dD      eD      fD      gD      hD      iD      jD      kD      lD      mD      nD      oD      pD      qD      rD      sD      tD      uD      vD      wD      xD      yD      zD      {D      |D      }D      ~D      D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D      ?D       E      E      E      E      E      E      E      E      E      	E      
E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E      E       E      !E      "E      #E      $E      %E      &E      'E      (E      )E      *E      +E      ,E      -E      .E      /E      0E      1E      2E      3E      4E      5E      6E      7E      8E      9E      :E      ;E      <E      =E      >E      ?E      @E      AE      BE      CE      DE      EE      FE      GE      HE      IE      JE      KE      LE      ME      NE      OE      PE      QE      RE      SE      TE      UE      VE      WE      XE      YE      ZE      [E      \E      ]E      ^E      _E      `E      aE      bE      cE      dE      eE      fE      gE      hE      iE      jE      kE      lE      mE      nE      oE      pE      qE      rE      sE      tE      uE      vE      wE      xE      yE      zE      {E      |E      }E      ~E      E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E      ?E       F      F      F      F      F      F      F      F      F      	F      
F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F      F       F      !F      "F      #F      $F      %F      &F      'F      (F      )F      *F      +F      ,F      -F      .F      /F      0F      1F      2F      3F      4F      5F      6F      7F      8F      9F      :F      ;F      <F      =F      >F      ?F      @F      AF      BF      CF      DF      EF      FF      GF      HF      IF      JF      KF      LF      MF      NF      OF      PF      QF      RF      SF      TF      UF      VF      WF      XF      YF      ZF      [F      \F      ]F      ^F      _F      `F      aF      bF      cF      dF      eF      fF      gF      hF      iF      jF      kF      lF      mF      nF      oF      pF      qF      rF      sF      tF      uF      vF      wF      xF      yF      zF      {F      |F      }F      ~F      F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F      ?F       G      G      G      G      G      G      G      G      G      	G      
G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G      G       G      !G      "G      #G      $G      %G      &G      'G      (G      )G      *G      +G      ,G      -G      .G      /G      0G      1G      2G      3G      4G      5G      6G      7G      8G      9G      :G      ;G      <G      =G      >G      ?G      @G      AG      BG      CG      DG      EG      FG      GG      HG      IG      JG      KG      LG      MG      NG      OG      PG      QG      RG      SG      TG      UG      VG      WG      XG      YG      ZG      [G      \G      ]G      ^G      _G      `G      aG      bG      cG      dG      eG      fG      gG      hG      iG      jG      kG      lG      mG      nG      oG      pG      qG      rG      sG      tG      uG      vG      wG      xG      yG      zG      {G      |G      }G      ~G      G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G      ?G       H      H      H      H      H      H      H      H      H      	H      
H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H      H       H      !H      "H      #H      $H      %H      &H      'H      (H      )H      *H      +H      ,H      -H      .H      /H      0H      1H      2H      3H      4H      5H      6H      7H      8H      9H      :H      ;H      <H      =H      >H      ?H      @H      AH      BH      CH      DH      EH      FH      GH      HH      IH      JH      KH      LH      MH      NH      OH      PH      QH      RH      SH      TH      UH      VH      WH      XH      YH      ZH      [H      \H      ]H      ^H      _H      `H      aH      bH      cH      dH      eH      fH      gH      hH      iH      jH      kH      lH      mH      nH      oH      pH      qH      rH      sH      tH      uH      vH      wH      xH      yH      zH      {H      |H      }H      ~H      H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H      ?H       I      I      I      I      I      I      I      I      I      	I      
I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I      I       I      !I      "I      #I      $I      %I      &I      'I      (I      )I      *I      +I      ,I      -I      .I      /I      0I      1I      2I      3I      4I      5I      6I      7I      8I      9I      :I      ;I      <I      =I      >I      ?I      @I      AI      BI      CI      DI      EI      FI      GI      HI      II      JI      KI      LI      MI      NI      OI      PI      QI      RI      SI      TI      UI      VI      WI      XI      YI      ZI      [I      \I      ]I      ^I      _I      `I      aI      bI      cI      dI      eI      fI      gI      hI      iI      jI      kI      lI      mI      nI      oI      pI      qI      rI      sI      tI      uI      vI      wI      xI      yI      zI      {I      |I      }I      ~I      I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I      ?I       J      J      J      J      J      J      J      J      J      	J      
J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J      J       J      !J      "J      #J      $J      %J      &J      'J      (J      )J      *J      +J      ,J      -J      .J      /J      0J      1J      2J      3J      4J      5J      6J      7J      8J      9J      :J      ;J      <J      =J      >J      ?J      @J      AJ      BJ      CJ      DJ      EJ      FJ      GJ      HJ      IJ      JJ      KJ      LJ      MJ      NJ      OJ      PJ      QJ      RJ      SJ      TJ      UJ      VJ      WJ      XJ      YJ      ZJ      [J      \J      ]J      ^J      _J      `J      aJ      bJ      cJ      dJ      eJ      fJ      gJ      hJ      iJ      jJ      kJ      lJ      mJ      nJ      oJ      pJ      qJ      rJ      sJ      tJ      uJ      vJ      wJ      xJ      yJ      zJ      {J      |J      }J      ~J      J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J      ?J       K      K      K      K      K      K      K      K      K      	K      
K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K      K       K      !K      "K      #K      $K      %K      &K      'K      (K      )K      *K      +K      ,K      -K      .K      /K      0K      1K      2K      3K      4K      5K      6K      7K      8K      9K      :K      ;K      <K      =K      >K      ?K      @K      AK      BK      CK      DK      EK      FK      GK      HK      IK      JK      KK      LK      MK      NK      OK      PK      QK      RK      SK      TK      UK      VK      WK      XK      YK      ZK      [K      \K      ]K      ^K      _K      `K      aK      bK      cK      dK      eK      fK      gK      hK      iK      jK      kK      lK      mK      nK      oK      pK      qK      rK      sK      tK      uK      vK      wK      xK      yK      zK      {K      |K      }K      ~K      K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K      ?K       L      L      L      L      L      L      L      L      L      	L      
L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L      L       L      !L      "L      #L      $L      %L      &L      'L      (L      )L      *L      +L      ,L      -L      .L      /L      0L      1L      2L      3L      4L      5L      6L      7L      8L      9L      :L      ;L      <L      =L      >L      ?L      @L      AL      BL      CL      DL      EL      FL      GL      HL      IL      JL      KL      LL      ML      NL      OL      PL      QL      RL      SL      TL      UL      VL      WL      XL      YL      ZL      [L      \L      ]L      ^L      _L      `L      aL      bL      cL      dL      eL      fL      gL      hL      iL      jL      kL      lL      mL      nL      oL      pL      qL      rL      sL      tL      uL      vL      wL      xL      yL      zL      {L      |L      }L      ~L      L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L      ?L       M      M      M      M      M      M      M      M      M      	M      
M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M      M       M      !M      "M      #M      $M      %M      &M      'M      (M      )M      *M      +M      ,M      -M      .M      /M      0M      1M      2M      3M      4M      5M      6M      7M      8M      9M      :M      ;M      <M      =M      >M      ?M      @M      AM      BM      CM      DM      EM      FM      GM      HM      IM      JM      KM      LM      MM      NM      OM      PM      QM      RM      SM      TM      UM      VM      WM      XM      YM      ZM      [M      \M      ]M      ^M      _M      `M      aM      bM      cM      dM      eM      fM      gM      hM      iM      jM      kM      lM      mM      nM      oM      pM      qM      rM      sM      tM      uM      vM      wM      xM      yM      zM      {M      |M      }M      ~M      M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M      ?M       N      N      N      N      N      N      N      N      N      	N      
N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      N      
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *#
fR
__inference_<lambda>_73138
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *#
fR
__inference_<lambda>_73143
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?-
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?,
value?,B?, B?,
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
;
	keras_api
_lookup_layer
_adapt_function*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

 kernel
!bias*
?
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias*
.
1
2
 3
!4
(5
)6*
.
0
1
 2
!3
(4
)5*
* 
?
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
/trace_0
0trace_1
1trace_2
2trace_3* 
6
3trace_0
4trace_1
5trace_2
6trace_3* 
* 
?
7iter
	8decay
9learning_rate
:momentum
;rho	rmsn	rmso	 rmsp	!rmsq	(rmsr	)rmss*

<serving_default* 
* 
7
=	keras_api
>lookup_table
?token_counts*

@trace_0* 

0
1*

0
1*
* 
?
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ftrace_0* 

Gtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

 0
!1*

 0
!1*
* 
?
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Mtrace_0* 

Ntrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

(0
)1*

(0
)1*
* 
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

Ttrace_0* 

Utrace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

V0
W1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
OI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
R
X_initializer
Y_create_resource
Z_initialize
[_destroy_resource* 
?
\_create_resource
]_initialize
^_destroy_resource><layer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
_	variables
`	keras_api
	atotal
	bcount*
H
c	variables
d	keras_api
	etotal
	fcount
g
_fn_kwargs*
* 

htrace_0* 

itrace_0* 

jtrace_0* 

ktrace_0* 

ltrace_0* 

mtrace_0* 

a0
b1*

_	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

e0
f1*

c	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
??
VARIABLE_VALUERMSprop/dense/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUERMSprop/dense/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/dense_1/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUERMSprop/dense_1/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/dense_2/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUERMSprop/dense_2/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_input_1
hash_tableConstConst_1Const_2dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_72783
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1total_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,RMSprop/dense/kernel/rms/Read/ReadVariableOp*RMSprop/dense/bias/rms/Read/ReadVariableOp.RMSprop/dense_1/kernel/rms/Read/ReadVariableOp,RMSprop/dense_1/bias/rms/Read/ReadVariableOp.RMSprop/dense_2/kernel/rms/Read/ReadVariableOp,RMSprop/dense_2/bias/rms/Read/ReadVariableOpConst_6*$
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_73243
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhoMutableHashTabletotal_1count_1totalcountRMSprop/dense/kernel/rmsRMSprop/dense/bias/rmsRMSprop/dense_1/kernel/rmsRMSprop/dense_1/bias/rmsRMSprop/dense_2/kernel/rmsRMSprop/dense_2/bias/rms*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_73319۔

?
?
%__inference_dense_layer_call_fn_73020

inputs
unknown:
??@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_72330o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
'__inference_dense_2_layer_call_fn_73060

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_72363o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
B__inference_dense_2_layer_call_and_return_conditional_losses_73070

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?z
?
@__inference_model_layer_call_and_return_conditional_losses_72370

inputsU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
dense_72331:
??@
dense_72333:@
dense_1_72348:@ 
dense_1_72350: 
dense_2_72364: 
dense_2_72366:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
,text_vectorization/StringNGrams/StringNGramsStringNGrams5text_vectorization/StringSplit/StringSplitV2:values:0`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*2
_output_shapes 
:?????????:?????????*
left_pad *
ngram_widths
*
	pad_width *
preserve_short_sequences( *
	right_pad *
	separator ?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringNGrams/StringNGrams:ngrams:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringNGrams/StringNGrams:ngrams:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
/text_vectorization/string_lookup/bincount/ShapeShape2text_vectorization/string_lookup/Identity:output:0*
T0	*
_output_shapes
:y
/text_vectorization/string_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.text_vectorization/string_lookup/bincount/ProdProd8text_vectorization/string_lookup/bincount/Shape:output:08text_vectorization/string_lookup/bincount/Const:output:0*
T0*
_output_shapes
: u
3text_vectorization/string_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
1text_vectorization/string_lookup/bincount/GreaterGreater7text_vectorization/string_lookup/bincount/Prod:output:0<text_vectorization/string_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
.text_vectorization/string_lookup/bincount/CastCast5text_vectorization/string_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: {
1text_vectorization/string_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
9text_vectorization/string_lookup/bincount/RaggedReduceMaxMax2text_vectorization/string_lookup/Identity:output:0:text_vectorization/string_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: q
/text_vectorization/string_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
-text_vectorization/string_lookup/bincount/addAddV2Btext_vectorization/string_lookup/bincount/RaggedReduceMax:output:08text_vectorization/string_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
-text_vectorization/string_lookup/bincount/mulMul2text_vectorization/string_lookup/bincount/Cast:y:01text_vectorization/string_lookup/bincount/add:z:0*
T0	*
_output_shapes
: w
3text_vectorization/string_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
valueB		 R???
1text_vectorization/string_lookup/bincount/MaximumMaximum<text_vectorization/string_lookup/bincount/minlength:output:01text_vectorization/string_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: w
3text_vectorization/string_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
valueB		 R???
1text_vectorization/string_lookup/bincount/MinimumMinimum<text_vectorization/string_lookup/bincount/maxlength:output:05text_vectorization/string_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: t
1text_vectorization/string_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB ?
8text_vectorization/string_lookup/bincount/RaggedBincountRaggedBincount<text_vectorization/StringNGrams/StringNGrams:ngrams_splits:02text_vectorization/string_lookup/Identity:output:05text_vectorization/string_lookup/bincount/Minimum:z:0:text_vectorization/string_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*)
_output_shapes
:???????????*
binary_output(?
dense/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization/string_lookup/bincount/RaggedBincount:output:0dense_72331dense_72333*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_72330?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_72348dense_1_72350*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_72347?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_72364dense_2_72366*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_72363w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
%__inference_model_layer_call_fn_72833

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:
??@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_72534o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
#__inference_signature_wrapper_72783
input_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:
??@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_72247o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?z
?
@__inference_model_layer_call_and_return_conditional_losses_72666
input_1U
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
dense_72650:
??@
dense_72652:@
dense_1_72655:@ 
dense_1_72657: 
dense_2_72660: 
dense_2_72662:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2_
text_vectorization/StringLowerStringLowerinput_1*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
,text_vectorization/StringNGrams/StringNGramsStringNGrams5text_vectorization/StringSplit/StringSplitV2:values:0`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*2
_output_shapes 
:?????????:?????????*
left_pad *
ngram_widths
*
	pad_width *
preserve_short_sequences( *
	right_pad *
	separator ?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringNGrams/StringNGrams:ngrams:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringNGrams/StringNGrams:ngrams:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
/text_vectorization/string_lookup/bincount/ShapeShape2text_vectorization/string_lookup/Identity:output:0*
T0	*
_output_shapes
:y
/text_vectorization/string_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.text_vectorization/string_lookup/bincount/ProdProd8text_vectorization/string_lookup/bincount/Shape:output:08text_vectorization/string_lookup/bincount/Const:output:0*
T0*
_output_shapes
: u
3text_vectorization/string_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
1text_vectorization/string_lookup/bincount/GreaterGreater7text_vectorization/string_lookup/bincount/Prod:output:0<text_vectorization/string_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
.text_vectorization/string_lookup/bincount/CastCast5text_vectorization/string_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: {
1text_vectorization/string_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
9text_vectorization/string_lookup/bincount/RaggedReduceMaxMax2text_vectorization/string_lookup/Identity:output:0:text_vectorization/string_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: q
/text_vectorization/string_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
-text_vectorization/string_lookup/bincount/addAddV2Btext_vectorization/string_lookup/bincount/RaggedReduceMax:output:08text_vectorization/string_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
-text_vectorization/string_lookup/bincount/mulMul2text_vectorization/string_lookup/bincount/Cast:y:01text_vectorization/string_lookup/bincount/add:z:0*
T0	*
_output_shapes
: w
3text_vectorization/string_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
valueB		 R???
1text_vectorization/string_lookup/bincount/MaximumMaximum<text_vectorization/string_lookup/bincount/minlength:output:01text_vectorization/string_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: w
3text_vectorization/string_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
valueB		 R???
1text_vectorization/string_lookup/bincount/MinimumMinimum<text_vectorization/string_lookup/bincount/maxlength:output:05text_vectorization/string_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: t
1text_vectorization/string_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB ?
8text_vectorization/string_lookup/bincount/RaggedBincountRaggedBincount<text_vectorization/StringNGrams/StringNGrams:ngrams_splits:02text_vectorization/string_lookup/Identity:output:05text_vectorization/string_lookup/bincount/Minimum:z:0:text_vectorization/string_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*)
_output_shapes
:???????????*
binary_output(?
dense/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization/string_lookup/bincount/RaggedBincount:output:0dense_72650dense_72652*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_72330?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_72655dense_1_72657*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_72347?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_72660dense_2_72662*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_72363w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
@__inference_dense_layer_call_and_return_conditional_losses_72330

inputs2
matmul_readvariableop_resource:
??@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
__inference_restore_fn_73130
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?

?
%__inference_model_layer_call_fn_72808

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:
??@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_72370o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
B__inference_dense_2_layer_call_and_return_conditional_losses_72363

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_72347

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_73031

inputs2
matmul_readvariableop_resource:
??@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?5
?	
__inference__traced_save_73243
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableopJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_rmsprop_dense_kernel_rms_read_readvariableop5
1savev2_rmsprop_dense_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_1_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_1_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_2_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_2_bias_rms_read_readvariableop
savev2_const_6

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B ?	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_rmsprop_dense_kernel_rms_read_readvariableop1savev2_rmsprop_dense_bias_rms_read_readvariableop5savev2_rmsprop_dense_1_kernel_rms_read_readvariableop3savev2_rmsprop_dense_1_bias_rms_read_readvariableop5savev2_rmsprop_dense_2_kernel_rms_read_readvariableop3savev2_rmsprop_dense_2_bias_rms_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *&
dtypes
2		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??@:@:@ : : :: : : : : ::: : : : :
??@:@:@ : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
?z
?
@__inference_model_layer_call_and_return_conditional_losses_72534

inputsU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
dense_72518:
??@
dense_72520:@
dense_1_72523:@ 
dense_1_72525: 
dense_2_72528: 
dense_2_72530:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
,text_vectorization/StringNGrams/StringNGramsStringNGrams5text_vectorization/StringSplit/StringSplitV2:values:0`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*2
_output_shapes 
:?????????:?????????*
left_pad *
ngram_widths
*
	pad_width *
preserve_short_sequences( *
	right_pad *
	separator ?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringNGrams/StringNGrams:ngrams:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringNGrams/StringNGrams:ngrams:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
/text_vectorization/string_lookup/bincount/ShapeShape2text_vectorization/string_lookup/Identity:output:0*
T0	*
_output_shapes
:y
/text_vectorization/string_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.text_vectorization/string_lookup/bincount/ProdProd8text_vectorization/string_lookup/bincount/Shape:output:08text_vectorization/string_lookup/bincount/Const:output:0*
T0*
_output_shapes
: u
3text_vectorization/string_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
1text_vectorization/string_lookup/bincount/GreaterGreater7text_vectorization/string_lookup/bincount/Prod:output:0<text_vectorization/string_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
.text_vectorization/string_lookup/bincount/CastCast5text_vectorization/string_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: {
1text_vectorization/string_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
9text_vectorization/string_lookup/bincount/RaggedReduceMaxMax2text_vectorization/string_lookup/Identity:output:0:text_vectorization/string_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: q
/text_vectorization/string_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
-text_vectorization/string_lookup/bincount/addAddV2Btext_vectorization/string_lookup/bincount/RaggedReduceMax:output:08text_vectorization/string_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
-text_vectorization/string_lookup/bincount/mulMul2text_vectorization/string_lookup/bincount/Cast:y:01text_vectorization/string_lookup/bincount/add:z:0*
T0	*
_output_shapes
: w
3text_vectorization/string_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
valueB		 R???
1text_vectorization/string_lookup/bincount/MaximumMaximum<text_vectorization/string_lookup/bincount/minlength:output:01text_vectorization/string_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: w
3text_vectorization/string_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
valueB		 R???
1text_vectorization/string_lookup/bincount/MinimumMinimum<text_vectorization/string_lookup/bincount/maxlength:output:05text_vectorization/string_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: t
1text_vectorization/string_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB ?
8text_vectorization/string_lookup/bincount/RaggedBincountRaggedBincount<text_vectorization/StringNGrams/StringNGrams:ngrams_splits:02text_vectorization/string_lookup/Identity:output:05text_vectorization/string_lookup/bincount/Minimum:z:0:text_vectorization/string_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*)
_output_shapes
:???????????*
binary_output(?
dense/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization/string_lookup/bincount/RaggedBincount:output:0dense_72518dense_72520*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_72330?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_72523dense_1_72525*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_72347?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_72528dense_2_72530*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_72363w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
,
__inference__destroyer_73103
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?[
?
!__inference__traced_restore_73319
file_prefix1
assignvariableop_dense_kernel:
??@+
assignvariableop_1_dense_bias:@3
!assignvariableop_2_dense_1_kernel:@ -
assignvariableop_3_dense_1_bias: 3
!assignvariableop_4_dense_2_kernel: -
assignvariableop_5_dense_2_bias:)
assignvariableop_6_rmsprop_iter:	 *
 assignvariableop_7_rmsprop_decay: 2
(assignvariableop_8_rmsprop_learning_rate: -
#assignvariableop_9_rmsprop_momentum: )
assignvariableop_10_rmsprop_rho: M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: #
assignvariableop_13_total: #
assignvariableop_14_count: @
,assignvariableop_15_rmsprop_dense_kernel_rms:
??@8
*assignvariableop_16_rmsprop_dense_bias_rms:@@
.assignvariableop_17_rmsprop_dense_1_kernel_rms:@ :
,assignvariableop_18_rmsprop_dense_1_bias_rms: @
.assignvariableop_19_rmsprop_dense_2_kernel_rms: :
,assignvariableop_20_rmsprop_dense_2_bias_rms:
identity_22??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?2MutableHashTable_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*t
_output_shapesb
`::::::::::::::::::::::::*&
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_rmsprop_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp assignvariableop_7_rmsprop_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp(assignvariableop_8_rmsprop_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_rmsprop_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_rmsprop_rhoIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:11RestoreV2:tensors:12*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 _
Identity_11IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp,assignvariableop_15_rmsprop_dense_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp*assignvariableop_16_rmsprop_dense_bias_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp.assignvariableop_17_rmsprop_dense_1_kernel_rmsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp,assignvariableop_18_rmsprop_dense_1_bias_rmsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp.assignvariableop_19_rmsprop_dense_2_kernel_rmsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp,assignvariableop_20_rmsprop_dense_2_bias_rmsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "#
identity_22Identity_22:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
?
?
__inference__initializer_730837
3key_value_init1316_lookuptableimportv2_table_handle/
+key_value_init1316_lookuptableimportv2_keys1
-key_value_init1316_lookuptableimportv2_values	
identity??&key_value_init1316/LookupTableImportV2?
&key_value_init1316/LookupTableImportV2LookupTableImportV23key_value_init1316_lookuptableimportv2_table_handle+key_value_init1316_lookuptableimportv2_keys-key_value_init1316_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: o
NoOpNoOp'^key_value_init1316/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: :??:??2P
&key_value_init1316/LookupTableImportV2&key_value_init1316/LookupTableImportV2:"

_output_shapes

:??:"

_output_shapes

:??
?
*
__inference_<lambda>_73143
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
%__inference_model_layer_call_fn_72393
input_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:
??@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_72370o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
:
__inference__creator_73075
identity??
hash_tablel

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1317*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
__inference_save_fn_73122
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::P
add/yConst*
_output_shapes
: *
dtype0*
valueB B
table-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: T
add_1/yConst*
_output_shapes
: *
dtype0*
valueB Btable-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
Ȍ
?
 __inference__wrapped_model_72247
input_1[
Wmodel_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle\
Xmodel_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	2
.model_text_vectorization_string_lookup_equal_y5
1model_text_vectorization_string_lookup_selectv2_t	>
*model_dense_matmul_readvariableop_resource:
??@9
+model_dense_biasadd_readvariableop_resource:@>
,model_dense_1_matmul_readvariableop_resource:@ ;
-model_dense_1_biasadd_readvariableop_resource: >
,model_dense_2_matmul_readvariableop_resource: ;
-model_dense_2_biasadd_readvariableop_resource:
identity??"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?$model/dense_1/BiasAdd/ReadVariableOp?#model/dense_1/MatMul/ReadVariableOp?$model/dense_2/BiasAdd/ReadVariableOp?#model/dense_2/MatMul/ReadVariableOp?Jmodel/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2e
$model/text_vectorization/StringLowerStringLowerinput_1*'
_output_shapes
:??????????
+model/text_vectorization/StaticRegexReplaceStaticRegexReplace-model/text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
 model/text_vectorization/SqueezeSqueeze4model/text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????k
*model/text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
2model/text_vectorization/StringSplit/StringSplitV2StringSplitV2)model/text_vectorization/Squeeze:output:03model/text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
8model/text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
:model/text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
:model/text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
2model/text_vectorization/StringSplit/strided_sliceStridedSlice<model/text_vectorization/StringSplit/StringSplitV2:indices:0Amodel/text_vectorization/StringSplit/strided_slice/stack:output:0Cmodel/text_vectorization/StringSplit/strided_slice/stack_1:output:0Cmodel/text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
:model/text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<model/text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<model/text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4model/text_vectorization/StringSplit/strided_slice_1StridedSlice:model/text_vectorization/StringSplit/StringSplitV2:shape:0Cmodel/text_vectorization/StringSplit/strided_slice_1/stack:output:0Emodel/text_vectorization/StringSplit/strided_slice_1/stack_1:output:0Emodel/text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
[model/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast;model/text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
]model/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast=model/text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
emodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape_model/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
emodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdnmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0nmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
imodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
gmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatermmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0rmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
dmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastkmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
gmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
cmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax_model/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0pmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
emodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
cmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2lmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0nmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
cmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulhmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0gmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
gmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumamodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
gmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumamodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0kmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
gmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
hmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount_model/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0kmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0pmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
bmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
]model/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumomodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0kmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
fmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
bmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
]model/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2omodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0cmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0kmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
2model/text_vectorization/StringNGrams/StringNGramsStringNGrams;model/text_vectorization/StringSplit/StringSplitV2:values:0fmodel/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*2
_output_shapes 
:?????????:?????????*
left_pad *
ngram_widths
*
	pad_width *
preserve_short_sequences( *
	right_pad *
	separator ?
Jmodel/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Wmodel_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle;model/text_vectorization/StringNGrams/StringNGrams:ngrams:0Xmodel_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,model/text_vectorization/string_lookup/EqualEqual;model/text_vectorization/StringNGrams/StringNGrams:ngrams:0.model_text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
/model/text_vectorization/string_lookup/SelectV2SelectV20model/text_vectorization/string_lookup/Equal:z:01model_text_vectorization_string_lookup_selectv2_tSmodel/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/model/text_vectorization/string_lookup/IdentityIdentity8model/text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
5model/text_vectorization/string_lookup/bincount/ShapeShape8model/text_vectorization/string_lookup/Identity:output:0*
T0	*
_output_shapes
:
5model/text_vectorization/string_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
4model/text_vectorization/string_lookup/bincount/ProdProd>model/text_vectorization/string_lookup/bincount/Shape:output:0>model/text_vectorization/string_lookup/bincount/Const:output:0*
T0*
_output_shapes
: {
9model/text_vectorization/string_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
7model/text_vectorization/string_lookup/bincount/GreaterGreater=model/text_vectorization/string_lookup/bincount/Prod:output:0Bmodel/text_vectorization/string_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
4model/text_vectorization/string_lookup/bincount/CastCast;model/text_vectorization/string_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: ?
7model/text_vectorization/string_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
?model/text_vectorization/string_lookup/bincount/RaggedReduceMaxMax8model/text_vectorization/string_lookup/Identity:output:0@model/text_vectorization/string_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: w
5model/text_vectorization/string_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
3model/text_vectorization/string_lookup/bincount/addAddV2Hmodel/text_vectorization/string_lookup/bincount/RaggedReduceMax:output:0>model/text_vectorization/string_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
3model/text_vectorization/string_lookup/bincount/mulMul8model/text_vectorization/string_lookup/bincount/Cast:y:07model/text_vectorization/string_lookup/bincount/add:z:0*
T0	*
_output_shapes
: }
9model/text_vectorization/string_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
valueB		 R???
7model/text_vectorization/string_lookup/bincount/MaximumMaximumBmodel/text_vectorization/string_lookup/bincount/minlength:output:07model/text_vectorization/string_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: }
9model/text_vectorization/string_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
valueB		 R???
7model/text_vectorization/string_lookup/bincount/MinimumMinimumBmodel/text_vectorization/string_lookup/bincount/maxlength:output:0;model/text_vectorization/string_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: z
7model/text_vectorization/string_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB ?
>model/text_vectorization/string_lookup/bincount/RaggedBincountRaggedBincountBmodel/text_vectorization/StringNGrams/StringNGrams:ngrams_splits:08model/text_vectorization/string_lookup/Identity:output:0;model/text_vectorization/string_lookup/bincount/Minimum:z:0@model/text_vectorization/string_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*)
_output_shapes
:???????????*
binary_output(?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
model/dense/MatMulMatMulGmodel/text_vectorization/string_lookup/bincount/RaggedBincount:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? l
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
model/dense_2/MatMulMatMul model/dense_1/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m
IdentityIdentitymodel/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOpK^model/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2?
Jmodel/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Jmodel/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
@__inference_model_layer_call_and_return_conditional_losses_72922

inputsU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	8
$dense_matmul_readvariableop_resource:
??@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource: 5
'dense_2_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
,text_vectorization/StringNGrams/StringNGramsStringNGrams5text_vectorization/StringSplit/StringSplitV2:values:0`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*2
_output_shapes 
:?????????:?????????*
left_pad *
ngram_widths
*
	pad_width *
preserve_short_sequences( *
	right_pad *
	separator ?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringNGrams/StringNGrams:ngrams:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringNGrams/StringNGrams:ngrams:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
/text_vectorization/string_lookup/bincount/ShapeShape2text_vectorization/string_lookup/Identity:output:0*
T0	*
_output_shapes
:y
/text_vectorization/string_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.text_vectorization/string_lookup/bincount/ProdProd8text_vectorization/string_lookup/bincount/Shape:output:08text_vectorization/string_lookup/bincount/Const:output:0*
T0*
_output_shapes
: u
3text_vectorization/string_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
1text_vectorization/string_lookup/bincount/GreaterGreater7text_vectorization/string_lookup/bincount/Prod:output:0<text_vectorization/string_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
.text_vectorization/string_lookup/bincount/CastCast5text_vectorization/string_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: {
1text_vectorization/string_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
9text_vectorization/string_lookup/bincount/RaggedReduceMaxMax2text_vectorization/string_lookup/Identity:output:0:text_vectorization/string_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: q
/text_vectorization/string_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
-text_vectorization/string_lookup/bincount/addAddV2Btext_vectorization/string_lookup/bincount/RaggedReduceMax:output:08text_vectorization/string_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
-text_vectorization/string_lookup/bincount/mulMul2text_vectorization/string_lookup/bincount/Cast:y:01text_vectorization/string_lookup/bincount/add:z:0*
T0	*
_output_shapes
: w
3text_vectorization/string_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
valueB		 R???
1text_vectorization/string_lookup/bincount/MaximumMaximum<text_vectorization/string_lookup/bincount/minlength:output:01text_vectorization/string_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: w
3text_vectorization/string_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
valueB		 R???
1text_vectorization/string_lookup/bincount/MinimumMinimum<text_vectorization/string_lookup/bincount/maxlength:output:05text_vectorization/string_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: t
1text_vectorization/string_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB ?
8text_vectorization/string_lookup/bincount/RaggedBincountRaggedBincount<text_vectorization/StringNGrams/StringNGrams:ngrams_splits:02text_vectorization/string_lookup/Identity:output:05text_vectorization/string_lookup/bincount/Minimum:z:0:text_vectorization/string_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*)
_output_shapes
:???????????*
binary_output(?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
dense/MatMulMatMulAtext_vectorization/string_lookup/bincount/RaggedBincount:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? `
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOpE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
.
__inference__initializer_73098
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference_<lambda>_731387
3key_value_init1316_lookuptableimportv2_table_handle/
+key_value_init1316_lookuptableimportv2_keys1
-key_value_init1316_lookuptableimportv2_values	
identity??&key_value_init1316/LookupTableImportV2?
&key_value_init1316/LookupTableImportV2LookupTableImportV23key_value_init1316_lookuptableimportv2_table_handle+key_value_init1316_lookuptableimportv2_keys-key_value_init1316_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: o
NoOpNoOp'^key_value_init1316/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: :??:??2P
&key_value_init1316/LookupTableImportV2&key_value_init1316/LookupTableImportV2:"

_output_shapes

:??:"

_output_shapes

:??
?
,
__inference__destroyer_73088
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
??
?
@__inference_model_layer_call_and_return_conditional_losses_73011

inputsU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	8
$dense_matmul_readvariableop_resource:
??@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource: 5
'dense_2_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
,text_vectorization/StringNGrams/StringNGramsStringNGrams5text_vectorization/StringSplit/StringSplitV2:values:0`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*2
_output_shapes 
:?????????:?????????*
left_pad *
ngram_widths
*
	pad_width *
preserve_short_sequences( *
	right_pad *
	separator ?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringNGrams/StringNGrams:ngrams:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringNGrams/StringNGrams:ngrams:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
/text_vectorization/string_lookup/bincount/ShapeShape2text_vectorization/string_lookup/Identity:output:0*
T0	*
_output_shapes
:y
/text_vectorization/string_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.text_vectorization/string_lookup/bincount/ProdProd8text_vectorization/string_lookup/bincount/Shape:output:08text_vectorization/string_lookup/bincount/Const:output:0*
T0*
_output_shapes
: u
3text_vectorization/string_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
1text_vectorization/string_lookup/bincount/GreaterGreater7text_vectorization/string_lookup/bincount/Prod:output:0<text_vectorization/string_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
.text_vectorization/string_lookup/bincount/CastCast5text_vectorization/string_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: {
1text_vectorization/string_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
9text_vectorization/string_lookup/bincount/RaggedReduceMaxMax2text_vectorization/string_lookup/Identity:output:0:text_vectorization/string_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: q
/text_vectorization/string_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
-text_vectorization/string_lookup/bincount/addAddV2Btext_vectorization/string_lookup/bincount/RaggedReduceMax:output:08text_vectorization/string_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
-text_vectorization/string_lookup/bincount/mulMul2text_vectorization/string_lookup/bincount/Cast:y:01text_vectorization/string_lookup/bincount/add:z:0*
T0	*
_output_shapes
: w
3text_vectorization/string_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
valueB		 R???
1text_vectorization/string_lookup/bincount/MaximumMaximum<text_vectorization/string_lookup/bincount/minlength:output:01text_vectorization/string_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: w
3text_vectorization/string_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
valueB		 R???
1text_vectorization/string_lookup/bincount/MinimumMinimum<text_vectorization/string_lookup/bincount/maxlength:output:05text_vectorization/string_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: t
1text_vectorization/string_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB ?
8text_vectorization/string_lookup/bincount/RaggedBincountRaggedBincount<text_vectorization/StringNGrams/StringNGrams:ngrams_splits:02text_vectorization/string_lookup/Identity:output:05text_vectorization/string_lookup/bincount/Minimum:z:0:text_vectorization/string_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*)
_output_shapes
:???????????*
binary_output(?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??@*
dtype0?
dense/MatMulMatMulAtext_vectorization/string_lookup/bincount/RaggedBincount:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? `
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOpE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?E
?
__inference_adapt_step_5503
iterator9
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2]
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:??????????
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
StringNGrams/StringNGramsStringNGrams"StringSplit/StringSplitV2:values:0MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*2
_output_shapes 
:?????????:?????????*
left_pad *
ngram_widths
*
	pad_width *
preserve_short_sequences( *
	right_pad *
	separator ?
UniqueWithCountsUniqueWithCounts"StringNGrams/StringNGrams:ngrams:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:

_output_shapes
: 
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_73051

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_72582
input_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:
??@
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_72534o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?z
?
@__inference_model_layer_call_and_return_conditional_losses_72750
input_1U
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
dense_72734:
??@
dense_72736:@
dense_1_72739:@ 
dense_1_72741: 
dense_2_72744: 
dense_2_72746:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2_
text_vectorization/StringLowerStringLowerinput_1*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
,text_vectorization/StringNGrams/StringNGramsStringNGrams5text_vectorization/StringSplit/StringSplitV2:values:0`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0*2
_output_shapes 
:?????????:?????????*
left_pad *
ngram_widths
*
	pad_width *
preserve_short_sequences( *
	right_pad *
	separator ?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringNGrams/StringNGrams:ngrams:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringNGrams/StringNGrams:ngrams:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
/text_vectorization/string_lookup/bincount/ShapeShape2text_vectorization/string_lookup/Identity:output:0*
T0	*
_output_shapes
:y
/text_vectorization/string_lookup/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
.text_vectorization/string_lookup/bincount/ProdProd8text_vectorization/string_lookup/bincount/Shape:output:08text_vectorization/string_lookup/bincount/Const:output:0*
T0*
_output_shapes
: u
3text_vectorization/string_lookup/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
1text_vectorization/string_lookup/bincount/GreaterGreater7text_vectorization/string_lookup/bincount/Prod:output:0<text_vectorization/string_lookup/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
.text_vectorization/string_lookup/bincount/CastCast5text_vectorization/string_lookup/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: {
1text_vectorization/string_lookup/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
9text_vectorization/string_lookup/bincount/RaggedReduceMaxMax2text_vectorization/string_lookup/Identity:output:0:text_vectorization/string_lookup/bincount/Const_1:output:0*
T0	*
_output_shapes
: q
/text_vectorization/string_lookup/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
-text_vectorization/string_lookup/bincount/addAddV2Btext_vectorization/string_lookup/bincount/RaggedReduceMax:output:08text_vectorization/string_lookup/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
-text_vectorization/string_lookup/bincount/mulMul2text_vectorization/string_lookup/bincount/Cast:y:01text_vectorization/string_lookup/bincount/add:z:0*
T0	*
_output_shapes
: w
3text_vectorization/string_lookup/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
valueB		 R???
1text_vectorization/string_lookup/bincount/MaximumMaximum<text_vectorization/string_lookup/bincount/minlength:output:01text_vectorization/string_lookup/bincount/mul:z:0*
T0	*
_output_shapes
: w
3text_vectorization/string_lookup/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
valueB		 R???
1text_vectorization/string_lookup/bincount/MinimumMinimum<text_vectorization/string_lookup/bincount/maxlength:output:05text_vectorization/string_lookup/bincount/Maximum:z:0*
T0	*
_output_shapes
: t
1text_vectorization/string_lookup/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB ?
8text_vectorization/string_lookup/bincount/RaggedBincountRaggedBincount<text_vectorization/StringNGrams/StringNGrams:ngrams_splits:02text_vectorization/string_lookup/Identity:output:05text_vectorization/string_lookup/bincount/Minimum:z:0:text_vectorization/string_lookup/bincount/Const_2:output:0*
T0*

Tidx0	*)
_output_shapes
:???????????*
binary_output(?
dense/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization/string_lookup/bincount/RaggedBincount:output:0dense_72734dense_72736*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_72330?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_72739dense_1_72741*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_72347?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_72744dense_2_72746*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_72363w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
F
__inference__creator_73093
identity: ??MutableHashTable|
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_7*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?
?
'__inference_dense_1_layer_call_fn_73040

inputs
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_72347o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????=
dense_22
StatefulPartitionedCall_1:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
P
	keras_api
_lookup_layer
_adapt_function"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

 kernel
!bias"
_tf_keras_layer
?
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias"
_tf_keras_layer
J
1
2
 3
!4
(5
)6"
trackable_list_wrapper
J
0
1
 2
!3
(4
)5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
/trace_0
0trace_1
1trace_2
2trace_32?
%__inference_model_layer_call_fn_72393
%__inference_model_layer_call_fn_72808
%__inference_model_layer_call_fn_72833
%__inference_model_layer_call_fn_72582?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z/trace_0z0trace_1z1trace_2z2trace_3
?
3trace_0
4trace_1
5trace_2
6trace_32?
@__inference_model_layer_call_and_return_conditional_losses_72922
@__inference_model_layer_call_and_return_conditional_losses_73011
@__inference_model_layer_call_and_return_conditional_losses_72666
@__inference_model_layer_call_and_return_conditional_losses_72750?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z3trace_0z4trace_1z5trace_2z6trace_3
?B?
 __inference__wrapped_model_72247input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
7iter
	8decay
9learning_rate
:momentum
;rho	rmsn	rmso	 rmsp	!rmsq	(rmsr	)rmss"
	optimizer
,
<serving_default"
signature_map
"
_generic_user_object
L
=	keras_api
>lookup_table
?token_counts"
_tf_keras_layer
?
@trace_02?
__inference_adapt_step_5503?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z@trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
Ftrace_02?
%__inference_dense_layer_call_fn_73020?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zFtrace_0
?
Gtrace_02?
@__inference_dense_layer_call_and_return_conditional_losses_73031?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zGtrace_0
 :
??@2dense/kernel
:@2
dense/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
Mtrace_02?
'__inference_dense_1_layer_call_fn_73040?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zMtrace_0
?
Ntrace_02?
B__inference_dense_1_layer_call_and_return_conditional_losses_73051?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zNtrace_0
 :@ 2dense_1/kernel
: 2dense_1/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
?
Ttrace_02?
'__inference_dense_2_layer_call_fn_73060?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zTtrace_0
?
Utrace_02?
B__inference_dense_2_layer_call_and_return_conditional_losses_73070?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zUtrace_0
 : 2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_model_layer_call_fn_72393input_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_model_layer_call_fn_72808inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_model_layer_call_fn_72833inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_model_layer_call_fn_72582input_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_72922inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_73011inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_72666input_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_72750input_1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
?B?
#__inference_signature_wrapper_72783input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
f
X_initializer
Y_create_resource
Z_initialize
[_destroy_resourceR jtf.StaticHashTable
J
\_create_resource
]_initialize
^_destroy_resourceR Z
 tu
?B?
__inference_adapt_step_5503iterator"?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_dense_layer_call_fn_73020inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
@__inference_dense_layer_call_and_return_conditional_losses_73031inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_dense_1_layer_call_fn_73040inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_1_layer_call_and_return_conditional_losses_73051inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_dense_2_layer_call_fn_73060inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_2_layer_call_and_return_conditional_losses_73070inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
N
_	variables
`	keras_api
	atotal
	bcount"
_tf_keras_metric
^
c	variables
d	keras_api
	etotal
	fcount
g
_fn_kwargs"
_tf_keras_metric
"
_generic_user_object
?
htrace_02?
__inference__creator_73075?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? zhtrace_0
?
itrace_02?
__inference__initializer_73083?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? zitrace_0
?
jtrace_02?
__inference__destroyer_73088?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? zjtrace_0
?
ktrace_02?
__inference__creator_73093?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? zktrace_0
?
ltrace_02?
__inference__initializer_73098?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? zltrace_0
?
mtrace_02?
__inference__destroyer_73103?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? zmtrace_0
.
a0
b1"
trackable_list_wrapper
-
_	variables"
_generic_user_object
:  (2total
:  (2count
.
e0
f1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
?B?
__inference__creator_73075"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_73083"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_73088"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_73093"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_73098"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_73103"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
*:(
??@2RMSprop/dense/kernel/rms
": @2RMSprop/dense/bias/rms
*:(@ 2RMSprop/dense_1/kernel/rms
$:" 2RMSprop/dense_1/bias/rms
*:( 2RMSprop/dense_2/kernel/rms
$:"2RMSprop/dense_2/bias/rms
?B?
__inference_save_fn_73122checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_73130restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant6
__inference__creator_73075?

? 
? "? 6
__inference__creator_73093?

? 
? "? 8
__inference__destroyer_73088?

? 
? "? 8
__inference__destroyer_73103?

? 
? "? ?
__inference__initializer_73083>z{?

? 
? "? :
__inference__initializer_73098?

? 
? "? ?
 __inference__wrapped_model_72247q
>vwx !()0?-
&?#
!?
input_1?????????
? "1?.
,
dense_2!?
dense_2?????????h
__inference_adapt_step_5503I?y??<
5?2
0?-?
??????????IteratorSpec 
? "
 ?
B__inference_dense_1_layer_call_and_return_conditional_losses_73051\ !/?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? z
'__inference_dense_1_layer_call_fn_73040O !/?,
%?"
 ?
inputs?????????@
? "?????????? ?
B__inference_dense_2_layer_call_and_return_conditional_losses_73070\()/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? z
'__inference_dense_2_layer_call_fn_73060O()/?,
%?"
 ?
inputs????????? 
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_73031^1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????@
? z
%__inference_dense_layer_call_fn_73020Q1?.
'?$
"?
inputs???????????
? "??????????@?
@__inference_model_layer_call_and_return_conditional_losses_72666m
>vwx !()8?5
.?+
!?
input_1?????????
p 

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_72750m
>vwx !()8?5
.?+
!?
input_1?????????
p

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_72922l
>vwx !()7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_73011l
>vwx !()7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
%__inference_model_layer_call_fn_72393`
>vwx !()8?5
.?+
!?
input_1?????????
p 

 
? "???????????
%__inference_model_layer_call_fn_72582`
>vwx !()8?5
.?+
!?
input_1?????????
p

 
? "???????????
%__inference_model_layer_call_fn_72808_
>vwx !()7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
%__inference_model_layer_call_fn_72833_
>vwx !()7?4
-?*
 ?
inputs?????????
p

 
? "??????????y
__inference_restore_fn_73130Y?K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_73122??&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
#__inference_signature_wrapper_72783|
>vwx !();?8
? 
1?.
,
input_1!?
input_1?????????"1?.
,
dense_2!?
dense_2?????????