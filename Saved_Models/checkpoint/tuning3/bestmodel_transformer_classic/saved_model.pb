ط"
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
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
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
;
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.1-0-g85c8b2a817f8??
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:( *
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
: *
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:  *
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
: *
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

: *
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:*
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
d
momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
momentum
]
momentum/Read/ReadVariableOpReadVariableOpmomentum*
_output_shapes
: *
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
?
1token_and_position_embedding/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *B
shared_name31token_and_position_embedding/embedding/embeddings
?
Etoken_and_position_embedding/embedding/embeddings/Read/ReadVariableOpReadVariableOp1token_and_position_embedding/embedding/embeddings*
_output_shapes

: *
dtype0
?
3token_and_position_embedding/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?i *D
shared_name53token_and_position_embedding/embedding_1/embeddings
?
Gtoken_and_position_embedding/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp3token_and_position_embedding/embedding_1/embeddings*
_output_shapes
:	?i *
dtype0
?
7transformer_block_1/multi_head_attention_1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_1/multi_head_attention_1/query/kernel
?
Ktransformer_block_1/multi_head_attention_1/query/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_1/multi_head_attention_1/query/kernel*"
_output_shapes
:  *
dtype0
?
5transformer_block_1/multi_head_attention_1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_1/multi_head_attention_1/query/bias
?
Itransformer_block_1/multi_head_attention_1/query/bias/Read/ReadVariableOpReadVariableOp5transformer_block_1/multi_head_attention_1/query/bias*
_output_shapes

: *
dtype0
?
5transformer_block_1/multi_head_attention_1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *F
shared_name75transformer_block_1/multi_head_attention_1/key/kernel
?
Itransformer_block_1/multi_head_attention_1/key/kernel/Read/ReadVariableOpReadVariableOp5transformer_block_1/multi_head_attention_1/key/kernel*"
_output_shapes
:  *
dtype0
?
3transformer_block_1/multi_head_attention_1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *D
shared_name53transformer_block_1/multi_head_attention_1/key/bias
?
Gtransformer_block_1/multi_head_attention_1/key/bias/Read/ReadVariableOpReadVariableOp3transformer_block_1/multi_head_attention_1/key/bias*
_output_shapes

: *
dtype0
?
7transformer_block_1/multi_head_attention_1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_1/multi_head_attention_1/value/kernel
?
Ktransformer_block_1/multi_head_attention_1/value/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_1/multi_head_attention_1/value/kernel*"
_output_shapes
:  *
dtype0
?
5transformer_block_1/multi_head_attention_1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_1/multi_head_attention_1/value/bias
?
Itransformer_block_1/multi_head_attention_1/value/bias/Read/ReadVariableOpReadVariableOp5transformer_block_1/multi_head_attention_1/value/bias*
_output_shapes

: *
dtype0
?
Btransformer_block_1/multi_head_attention_1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBtransformer_block_1/multi_head_attention_1/attention_output/kernel
?
Vtransformer_block_1/multi_head_attention_1/attention_output/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_1/multi_head_attention_1/attention_output/kernel*"
_output_shapes
:  *
dtype0
?
@transformer_block_1/multi_head_attention_1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_1/multi_head_attention_1/attention_output/bias
?
Ttransformer_block_1/multi_head_attention_1/attention_output/bias/Read/ReadVariableOpReadVariableOp@transformer_block_1/multi_head_attention_1/attention_output/bias*
_output_shapes
: *
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:  *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
: *
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:  *
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
: *
dtype0
?
/transformer_block_1/layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_1/layer_normalization_2/gamma
?
Ctransformer_block_1/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_1/layer_normalization_2/gamma*
_output_shapes
: *
dtype0
?
.transformer_block_1/layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_1/layer_normalization_2/beta
?
Btransformer_block_1/layer_normalization_2/beta/Read/ReadVariableOpReadVariableOp.transformer_block_1/layer_normalization_2/beta*
_output_shapes
: *
dtype0
?
/transformer_block_1/layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_1/layer_normalization_3/gamma
?
Ctransformer_block_1/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_1/layer_normalization_3/gamma*
_output_shapes
: *
dtype0
?
.transformer_block_1/layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_1/layer_normalization_3/beta
?
Btransformer_block_1/layer_normalization_3/beta/Read/ReadVariableOpReadVariableOp.transformer_block_1/layer_normalization_3/beta*
_output_shapes
: *
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
?
SGD/dense_4/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *,
shared_nameSGD/dense_4/kernel/momentum
?
/SGD/dense_4/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_4/kernel/momentum*
_output_shapes

:( *
dtype0
?
SGD/dense_4/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameSGD/dense_4/bias/momentum
?
-SGD/dense_4/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_4/bias/momentum*
_output_shapes
: *
dtype0
?
SGD/dense_5/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *,
shared_nameSGD/dense_5/kernel/momentum
?
/SGD/dense_5/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_5/kernel/momentum*
_output_shapes

:  *
dtype0
?
SGD/dense_5/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameSGD/dense_5/bias/momentum
?
-SGD/dense_5/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_5/bias/momentum*
_output_shapes
: *
dtype0
?
SGD/dense_6/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *,
shared_nameSGD/dense_6/kernel/momentum
?
/SGD/dense_6/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_6/kernel/momentum*
_output_shapes

: *
dtype0
?
SGD/dense_6/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameSGD/dense_6/bias/momentum
?
-SGD/dense_6/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_6/bias/momentum*
_output_shapes
:*
dtype0
?
>SGD/token_and_position_embedding/embedding/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *O
shared_name@>SGD/token_and_position_embedding/embedding/embeddings/momentum
?
RSGD/token_and_position_embedding/embedding/embeddings/momentum/Read/ReadVariableOpReadVariableOp>SGD/token_and_position_embedding/embedding/embeddings/momentum*
_output_shapes

: *
dtype0
?
@SGD/token_and_position_embedding/embedding_1/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?i *Q
shared_nameB@SGD/token_and_position_embedding/embedding_1/embeddings/momentum
?
TSGD/token_and_position_embedding/embedding_1/embeddings/momentum/Read/ReadVariableOpReadVariableOp@SGD/token_and_position_embedding/embedding_1/embeddings/momentum*
_output_shapes
:	?i *
dtype0
?
DSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentum
?
XSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentum*"
_output_shapes
:  *
dtype0
?
BSGD/transformer_block_1/multi_head_attention_1/query/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_1/multi_head_attention_1/query/bias/momentum
?
VSGD/transformer_block_1/multi_head_attention_1/query/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_1/multi_head_attention_1/query/bias/momentum*
_output_shapes

: *
dtype0
?
BSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentum
?
VSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentum*"
_output_shapes
:  *
dtype0
?
@SGD/transformer_block_1/multi_head_attention_1/key/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *Q
shared_nameB@SGD/transformer_block_1/multi_head_attention_1/key/bias/momentum
?
TSGD/transformer_block_1/multi_head_attention_1/key/bias/momentum/Read/ReadVariableOpReadVariableOp@SGD/transformer_block_1/multi_head_attention_1/key/bias/momentum*
_output_shapes

: *
dtype0
?
DSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentum
?
XSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentum*"
_output_shapes
:  *
dtype0
?
BSGD/transformer_block_1/multi_head_attention_1/value/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_1/multi_head_attention_1/value/bias/momentum
?
VSGD/transformer_block_1/multi_head_attention_1/value/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_1/multi_head_attention_1/value/bias/momentum*
_output_shapes

: *
dtype0
?
OSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *`
shared_nameQOSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentum
?
cSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentum/Read/ReadVariableOpReadVariableOpOSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentum*"
_output_shapes
:  *
dtype0
?
MSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *^
shared_nameOMSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentum
?
aSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentum/Read/ReadVariableOpReadVariableOpMSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentum*
_output_shapes
: *
dtype0
?
SGD/dense_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *,
shared_nameSGD/dense_2/kernel/momentum
?
/SGD/dense_2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_2/kernel/momentum*
_output_shapes

:  *
dtype0
?
SGD/dense_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameSGD/dense_2/bias/momentum
?
-SGD/dense_2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_2/bias/momentum*
_output_shapes
: *
dtype0
?
SGD/dense_3/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *,
shared_nameSGD/dense_3/kernel/momentum
?
/SGD/dense_3/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_3/kernel/momentum*
_output_shapes

:  *
dtype0
?
SGD/dense_3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameSGD/dense_3/bias/momentum
?
-SGD/dense_3/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_3/bias/momentum*
_output_shapes
: *
dtype0
?
<SGD/transformer_block_1/layer_normalization_2/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_1/layer_normalization_2/gamma/momentum
?
PSGD/transformer_block_1/layer_normalization_2/gamma/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_1/layer_normalization_2/gamma/momentum*
_output_shapes
: *
dtype0
?
;SGD/transformer_block_1/layer_normalization_2/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;SGD/transformer_block_1/layer_normalization_2/beta/momentum
?
OSGD/transformer_block_1/layer_normalization_2/beta/momentum/Read/ReadVariableOpReadVariableOp;SGD/transformer_block_1/layer_normalization_2/beta/momentum*
_output_shapes
: *
dtype0
?
<SGD/transformer_block_1/layer_normalization_3/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_1/layer_normalization_3/gamma/momentum
?
PSGD/transformer_block_1/layer_normalization_3/gamma/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_1/layer_normalization_3/gamma/momentum*
_output_shapes
: *
dtype0
?
;SGD/transformer_block_1/layer_normalization_3/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;SGD/transformer_block_1/layer_normalization_3/beta/momentum
?
OSGD/transformer_block_1/layer_normalization_3/beta/momentum/Read/ReadVariableOpReadVariableOp;SGD/transformer_block_1/layer_normalization_3/beta/momentum*
_output_shapes
: *
dtype0

NoOpNoOp
ˊ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
n
	token_emb
pos_emb
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
?
att
ffn

layernorm1
 
layernorm2
!dropout1
"dropout2
#trainable_variables
$	variables
%regularization_losses
&	keras_api
R
'trainable_variables
(	variables
)regularization_losses
*	keras_api
 
R
+trainable_variables
,	variables
-regularization_losses
.	keras_api
h

/kernel
0bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
R
5trainable_variables
6	variables
7regularization_losses
8	keras_api
h

9kernel
:bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
R
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
h

Ckernel
Dbias
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
?
	Idecay
Jlearning_rate
Kmomentum
Liter/momentum?0momentum?9momentum?:momentum?Cmomentum?Dmomentum?Mmomentum?Nmomentum?Omomentum?Pmomentum?Qmomentum?Rmomentum?Smomentum?Tmomentum?Umomentum?Vmomentum?Wmomentum?Xmomentum?Ymomentum?Zmomentum?[momentum?\momentum?]momentum?^momentum?
?
M0
N1
O2
P3
Q4
R5
S6
T7
U8
V9
W10
X11
Y12
Z13
[14
\15
]16
^17
/18
019
920
:21
C22
D23
?
M0
N1
O2
P3
Q4
R5
S6
T7
U8
V9
W10
X11
Y12
Z13
[14
\15
]16
^17
/18
019
920
:21
C22
D23
 
?
_non_trainable_variables
`metrics
alayer_metrics
trainable_variables
blayer_regularization_losses

clayers
	variables
regularization_losses
 
b
M
embeddings
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
b
N
embeddings
htrainable_variables
i	variables
jregularization_losses
k	keras_api

M0
N1

M0
N1
 
?
lnon_trainable_variables
mmetrics
nlayer_metrics
trainable_variables
olayer_regularization_losses

players
	variables
regularization_losses
 
 
 
?
qnon_trainable_variables
rmetrics
slayer_metrics
trainable_variables
tlayer_regularization_losses

ulayers
	variables
regularization_losses
?
v_query_dense
w
_key_dense
x_value_dense
y_softmax
z_dropout_layer
{_output_dense
|trainable_variables
}	variables
~regularization_losses
	keras_api
?
?layer_with_weights-0
?layer-0
?layer_with_weights-1
?layer-1
?trainable_variables
?	variables
?regularization_losses
?	keras_api
v
	?axis
	[gamma
\beta
?trainable_variables
?	variables
?regularization_losses
?	keras_api
v
	?axis
	]gamma
^beta
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
v
O0
P1
Q2
R3
S4
T5
U6
V7
W8
X9
Y10
Z11
[12
\13
]14
^15
v
O0
P1
Q2
R3
S4
T5
U6
V7
W8
X9
Y10
Z11
[12
\13
]14
^15
 
?
?non_trainable_variables
?metrics
?layer_metrics
#trainable_variables
 ?layer_regularization_losses
?layers
$	variables
%regularization_losses
 
 
 
?
?non_trainable_variables
?metrics
?layer_metrics
'trainable_variables
 ?layer_regularization_losses
?layers
(	variables
)regularization_losses
 
 
 
?
?non_trainable_variables
?metrics
?layer_metrics
+trainable_variables
 ?layer_regularization_losses
?layers
,	variables
-regularization_losses
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01

/0
01
 
?
?non_trainable_variables
?metrics
?layer_metrics
1trainable_variables
 ?layer_regularization_losses
?layers
2	variables
3regularization_losses
 
 
 
?
?non_trainable_variables
?metrics
?layer_metrics
5trainable_variables
 ?layer_regularization_losses
?layers
6	variables
7regularization_losses
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1

90
:1
 
?
?non_trainable_variables
?metrics
?layer_metrics
;trainable_variables
 ?layer_regularization_losses
?layers
<	variables
=regularization_losses
 
 
 
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
@	variables
Aregularization_losses
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

C0
D1
 
?
?non_trainable_variables
?metrics
?layer_metrics
Etrainable_variables
 ?layer_regularization_losses
?layers
F	variables
Gregularization_losses
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE1token_and_position_embedding/embedding/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE3token_and_position_embedding/embedding_1/embeddings0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE7transformer_block_1/multi_head_attention_1/query/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE5transformer_block_1/multi_head_attention_1/query/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE5transformer_block_1/multi_head_attention_1/key/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE3transformer_block_1/multi_head_attention_1/key/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE7transformer_block_1/multi_head_attention_1/value/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE5transformer_block_1/multi_head_attention_1/value/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBtransformer_block_1/multi_head_attention_1/attention_output/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@transformer_block_1/multi_head_attention_1/attention_output/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_2/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_2/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_3/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_3/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/transformer_block_1/layer_normalization_2/gamma1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE.transformer_block_1/layer_normalization_2/beta1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/transformer_block_1/layer_normalization_3/gamma1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE.transformer_block_1/layer_normalization_3/beta1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
 

?0
 
 
V
0
1
2
3
4
5
6
7
	8

9
10
11

M0

M0
 
?
?non_trainable_variables
?metrics
?layer_metrics
dtrainable_variables
 ?layer_regularization_losses
?layers
e	variables
fregularization_losses

N0

N0
 
?
?non_trainable_variables
?metrics
?layer_metrics
htrainable_variables
 ?layer_regularization_losses
?layers
i	variables
jregularization_losses
 
 
 
 

0
1
 
 
 
 
 
?
?partial_output_shape
?full_output_shape

Okernel
Pbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?partial_output_shape
?full_output_shape

Qkernel
Rbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?partial_output_shape
?full_output_shape

Skernel
Tbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?partial_output_shape
?full_output_shape

Ukernel
Vbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
8
O0
P1
Q2
R3
S4
T5
U6
V7
8
O0
P1
Q2
R3
S4
T5
U6
V7
 
?
?non_trainable_variables
?metrics
?layer_metrics
|trainable_variables
 ?layer_regularization_losses
?layers
}	variables
~regularization_losses
l

Wkernel
Xbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
l

Ykernel
Zbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api

W0
X1
Y2
Z3

W0
X1
Y2
Z3
 
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
 

[0
\1

[0
\1
 
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
 

]0
^1

]0
^1
 
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
 
 
 
 
*
0
1
2
 3
!4
"5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 

O0
P1

O0
P1
 
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
 
 

Q0
R1

Q0
R1
 
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
 
 

S0
T1

S0
T1
 
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
 
 

U0
V1

U0
V1
 
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
 
 
 
 
*
v0
w1
x2
y3
z4
{5

W0
X1

W0
X1
 
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses

Y0
Z1

Y0
Z1
 
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
??
VARIABLE_VALUESGD/dense_4/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_4/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_5/kernel/momentumYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_5/bias/momentumWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_6/kernel/momentumYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_6/bias/momentumWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>SGD/token_and_position_embedding/embedding/embeddings/momentumStrainable_variables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@SGD/token_and_position_embedding/embedding_1/embeddings/momentumStrainable_variables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentumStrainable_variables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBSGD/transformer_block_1/multi_head_attention_1/query/bias/momentumStrainable_variables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentumStrainable_variables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@SGD/transformer_block_1/multi_head_attention_1/key/bias/momentumStrainable_variables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentumStrainable_variables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBSGD/transformer_block_1/multi_head_attention_1/value/bias/momentumStrainable_variables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEOSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentumStrainable_variables/8/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEMSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentumStrainable_variables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_2/kernel/momentumTtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_2/bias/momentumTtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_3/kernel/momentumTtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUESGD/dense_3/bias/momentumTtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE<SGD/transformer_block_1/layer_normalization_2/gamma/momentumTtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE;SGD/transformer_block_1/layer_normalization_2/beta/momentumTtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE<SGD/transformer_block_1/layer_normalization_3/gamma/momentumTtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE;SGD/transformer_block_1/layer_normalization_3/beta/momentumTtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????i*
dtype0*
shape:??????????i
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_23token_and_position_embedding/embedding_1/embeddings1token_and_position_embedding/embedding/embeddings7transformer_block_1/multi_head_attention_1/query/kernel5transformer_block_1/multi_head_attention_1/query/bias5transformer_block_1/multi_head_attention_1/key/kernel3transformer_block_1/multi_head_attention_1/key/bias7transformer_block_1/multi_head_attention_1/value/kernel5transformer_block_1/multi_head_attention_1/value/biasBtransformer_block_1/multi_head_attention_1/attention_output/kernel@transformer_block_1/multi_head_attention_1/attention_output/bias/transformer_block_1/layer_normalization_2/gamma.transformer_block_1/layer_normalization_2/betadense_2/kerneldense_2/biasdense_3/kerneldense_3/bias/transformer_block_1/layer_normalization_3/gamma.transformer_block_1/layer_normalization_3/betadense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/bias*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_89447
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpEtoken_and_position_embedding/embedding/embeddings/Read/ReadVariableOpGtoken_and_position_embedding/embedding_1/embeddings/Read/ReadVariableOpKtransformer_block_1/multi_head_attention_1/query/kernel/Read/ReadVariableOpItransformer_block_1/multi_head_attention_1/query/bias/Read/ReadVariableOpItransformer_block_1/multi_head_attention_1/key/kernel/Read/ReadVariableOpGtransformer_block_1/multi_head_attention_1/key/bias/Read/ReadVariableOpKtransformer_block_1/multi_head_attention_1/value/kernel/Read/ReadVariableOpItransformer_block_1/multi_head_attention_1/value/bias/Read/ReadVariableOpVtransformer_block_1/multi_head_attention_1/attention_output/kernel/Read/ReadVariableOpTtransformer_block_1/multi_head_attention_1/attention_output/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpCtransformer_block_1/layer_normalization_2/gamma/Read/ReadVariableOpBtransformer_block_1/layer_normalization_2/beta/Read/ReadVariableOpCtransformer_block_1/layer_normalization_3/gamma/Read/ReadVariableOpBtransformer_block_1/layer_normalization_3/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp/SGD/dense_4/kernel/momentum/Read/ReadVariableOp-SGD/dense_4/bias/momentum/Read/ReadVariableOp/SGD/dense_5/kernel/momentum/Read/ReadVariableOp-SGD/dense_5/bias/momentum/Read/ReadVariableOp/SGD/dense_6/kernel/momentum/Read/ReadVariableOp-SGD/dense_6/bias/momentum/Read/ReadVariableOpRSGD/token_and_position_embedding/embedding/embeddings/momentum/Read/ReadVariableOpTSGD/token_and_position_embedding/embedding_1/embeddings/momentum/Read/ReadVariableOpXSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_1/multi_head_attention_1/query/bias/momentum/Read/ReadVariableOpVSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentum/Read/ReadVariableOpTSGD/transformer_block_1/multi_head_attention_1/key/bias/momentum/Read/ReadVariableOpXSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_1/multi_head_attention_1/value/bias/momentum/Read/ReadVariableOpcSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentum/Read/ReadVariableOpaSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentum/Read/ReadVariableOp/SGD/dense_2/kernel/momentum/Read/ReadVariableOp-SGD/dense_2/bias/momentum/Read/ReadVariableOp/SGD/dense_3/kernel/momentum/Read/ReadVariableOp-SGD/dense_3/bias/momentum/Read/ReadVariableOpPSGD/transformer_block_1/layer_normalization_2/gamma/momentum/Read/ReadVariableOpOSGD/transformer_block_1/layer_normalization_2/beta/momentum/Read/ReadVariableOpPSGD/transformer_block_1/layer_normalization_3/gamma/momentum/Read/ReadVariableOpOSGD/transformer_block_1/layer_normalization_3/beta/momentum/Read/ReadVariableOpConst*C
Tin<
:28	*
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
__inference__traced_save_90881
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdecaylearning_ratemomentumSGD/iter1token_and_position_embedding/embedding/embeddings3token_and_position_embedding/embedding_1/embeddings7transformer_block_1/multi_head_attention_1/query/kernel5transformer_block_1/multi_head_attention_1/query/bias5transformer_block_1/multi_head_attention_1/key/kernel3transformer_block_1/multi_head_attention_1/key/bias7transformer_block_1/multi_head_attention_1/value/kernel5transformer_block_1/multi_head_attention_1/value/biasBtransformer_block_1/multi_head_attention_1/attention_output/kernel@transformer_block_1/multi_head_attention_1/attention_output/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias/transformer_block_1/layer_normalization_2/gamma.transformer_block_1/layer_normalization_2/beta/transformer_block_1/layer_normalization_3/gamma.transformer_block_1/layer_normalization_3/betatotalcountSGD/dense_4/kernel/momentumSGD/dense_4/bias/momentumSGD/dense_5/kernel/momentumSGD/dense_5/bias/momentumSGD/dense_6/kernel/momentumSGD/dense_6/bias/momentum>SGD/token_and_position_embedding/embedding/embeddings/momentum@SGD/token_and_position_embedding/embedding_1/embeddings/momentumDSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentumBSGD/transformer_block_1/multi_head_attention_1/query/bias/momentumBSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentum@SGD/transformer_block_1/multi_head_attention_1/key/bias/momentumDSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentumBSGD/transformer_block_1/multi_head_attention_1/value/bias/momentumOSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentumMSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentumSGD/dense_2/kernel/momentumSGD/dense_2/bias/momentumSGD/dense_3/kernel/momentumSGD/dense_3/bias/momentum<SGD/transformer_block_1/layer_normalization_2/gamma/momentum;SGD/transformer_block_1/layer_normalization_2/beta/momentum<SGD/transformer_block_1/layer_normalization_3/gamma/momentum;SGD/transformer_block_1/layer_normalization_3/beta/momentum*B
Tin;
927*
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
!__inference__traced_restore_91053Î
?
r
F__inference_concatenate_layer_call_and_return_conditional_losses_90357
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????(2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :?????????:Q M
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
??
?
@__inference_model_layer_call_and_return_conditional_losses_89838
inputs_0
inputs_1C
?token_and_position_embedding_embedding_1_embedding_lookup_89672A
=token_and_position_embedding_embedding_embedding_lookup_89678Z
Vtransformer_block_1_multi_head_attention_1_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_1_multi_head_attention_1_query_add_readvariableop_resourceX
Ttransformer_block_1_multi_head_attention_1_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_1_multi_head_attention_1_key_add_readvariableop_resourceZ
Vtransformer_block_1_multi_head_attention_1_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_1_multi_head_attention_1_value_add_readvariableop_resourcee
atransformer_block_1_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_1_multi_head_attention_1_attention_output_add_readvariableop_resourceS
Otransformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_1_layer_normalization_2_batchnorm_readvariableop_resourceN
Jtransformer_block_1_sequential_1_dense_2_tensordot_readvariableop_resourceL
Htransformer_block_1_sequential_1_dense_2_biasadd_readvariableop_resourceN
Jtransformer_block_1_sequential_1_dense_3_tensordot_readvariableop_resourceL
Htransformer_block_1_sequential_1_dense_3_biasadd_readvariableop_resourceS
Otransformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?7token_and_position_embedding/embedding/embedding_lookup?9token_and_position_embedding/embedding_1/embedding_lookup?Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp?Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp?Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp?Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp?Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp?Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp?Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp?Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp?Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp?Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp?Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp??transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp?Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp??transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp?Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp?
"token_and_position_embedding/ShapeShapeinputs_0*
T0*
_output_shapes
:2$
"token_and_position_embedding/Shape?
0token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????22
0token_and_position_embedding/strided_slice/stack?
2token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 24
2token_and_position_embedding/strided_slice/stack_1?
2token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2token_and_position_embedding/strided_slice/stack_2?
*token_and_position_embedding/strided_sliceStridedSlice+token_and_position_embedding/Shape:output:09token_and_position_embedding/strided_slice/stack:output:0;token_and_position_embedding/strided_slice/stack_1:output:0;token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*token_and_position_embedding/strided_slice?
(token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2*
(token_and_position_embedding/range/start?
(token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2*
(token_and_position_embedding/range/delta?
"token_and_position_embedding/rangeRange1token_and_position_embedding/range/start:output:03token_and_position_embedding/strided_slice:output:01token_and_position_embedding/range/delta:output:0*#
_output_shapes
:?????????2$
"token_and_position_embedding/range?
9token_and_position_embedding/embedding_1/embedding_lookupResourceGather?token_and_position_embedding_embedding_1_embedding_lookup_89672+token_and_position_embedding/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*R
_classH
FDloc:@token_and_position_embedding/embedding_1/embedding_lookup/89672*'
_output_shapes
:????????? *
dtype02;
9token_and_position_embedding/embedding_1/embedding_lookup?
Btoken_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityBtoken_and_position_embedding/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*R
_classH
FDloc:@token_and_position_embedding/embedding_1/embedding_lookup/89672*'
_output_shapes
:????????? 2D
Btoken_and_position_embedding/embedding_1/embedding_lookup/Identity?
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityKtoken_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2F
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1?
+token_and_position_embedding/embedding/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:??????????i2-
+token_and_position_embedding/embedding/Cast?
7token_and_position_embedding/embedding/embedding_lookupResourceGather=token_and_position_embedding_embedding_embedding_lookup_89678/token_and_position_embedding/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@token_and_position_embedding/embedding/embedding_lookup/89678*,
_output_shapes
:??????????i *
dtype029
7token_and_position_embedding/embedding/embedding_lookup?
@token_and_position_embedding/embedding/embedding_lookup/IdentityIdentity@token_and_position_embedding/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@token_and_position_embedding/embedding/embedding_lookup/89678*,
_output_shapes
:??????????i 2B
@token_and_position_embedding/embedding/embedding_lookup/Identity?
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityItoken_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????i 2D
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1?
 token_and_position_embedding/addAddV2Ktoken_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Mtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????i 2"
 token_and_position_embedding/add?
 average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 average_pooling1d/ExpandDims/dim?
average_pooling1d/ExpandDims
ExpandDims$token_and_position_embedding/add:z:0)average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????i 2
average_pooling1d/ExpandDims?
average_pooling1d/AvgPoolAvgPool%average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:?????????- *
ksize	
?*
paddingVALID*
strides	
?2
average_pooling1d/AvgPool?
average_pooling1d/SqueezeSqueeze"average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:?????????- *
squeeze_dims
2
average_pooling1d/Squeeze?
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_1_multi_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?
>transformer_block_1/multi_head_attention_1/query/einsum/EinsumEinsum"average_pooling1d/Squeeze:output:0Utransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2@
>transformer_block_1/multi_head_attention_1/query/einsum/Einsum?
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOpReadVariableOpLtransformer_block_1_multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp?
4transformer_block_1/multi_head_attention_1/query/addAddV2Gtransformer_block_1/multi_head_attention_1/query/einsum/Einsum:output:0Ktransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 26
4transformer_block_1/multi_head_attention_1/query/add?
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_1_multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp?
<transformer_block_1/multi_head_attention_1/key/einsum/EinsumEinsum"average_pooling1d/Squeeze:output:0Stransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2>
<transformer_block_1/multi_head_attention_1/key/einsum/Einsum?
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOpReadVariableOpJtransformer_block_1_multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp?
2transformer_block_1/multi_head_attention_1/key/addAddV2Etransformer_block_1/multi_head_attention_1/key/einsum/Einsum:output:0Itransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 24
2transformer_block_1/multi_head_attention_1/key/add?
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_1_multi_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp?
>transformer_block_1/multi_head_attention_1/value/einsum/EinsumEinsum"average_pooling1d/Squeeze:output:0Utransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2@
>transformer_block_1/multi_head_attention_1/value/einsum/Einsum?
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOpReadVariableOpLtransformer_block_1_multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp?
4transformer_block_1/multi_head_attention_1/value/addAddV2Gtransformer_block_1/multi_head_attention_1/value/einsum/Einsum:output:0Ktransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 26
4transformer_block_1/multi_head_attention_1/value/add?
0transformer_block_1/multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>22
0transformer_block_1/multi_head_attention_1/Mul/y?
.transformer_block_1/multi_head_attention_1/MulMul8transformer_block_1/multi_head_attention_1/query/add:z:09transformer_block_1/multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:?????????- 20
.transformer_block_1/multi_head_attention_1/Mul?
8transformer_block_1/multi_head_attention_1/einsum/EinsumEinsum6transformer_block_1/multi_head_attention_1/key/add:z:02transformer_block_1/multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:?????????--*
equationaecd,abcd->acbe2:
8transformer_block_1/multi_head_attention_1/einsum/Einsum?
:transformer_block_1/multi_head_attention_1/softmax/SoftmaxSoftmaxAtransformer_block_1/multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????--2<
:transformer_block_1/multi_head_attention_1/softmax/Softmax?
;transformer_block_1/multi_head_attention_1/dropout/IdentityIdentityDtransformer_block_1/multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????--2=
;transformer_block_1/multi_head_attention_1/dropout/Identity?
:transformer_block_1/multi_head_attention_1/einsum_1/EinsumEinsumDtransformer_block_1/multi_head_attention_1/dropout/Identity:output:08transformer_block_1/multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:?????????- *
equationacbe,aecd->abcd2<
:transformer_block_1/multi_head_attention_1/einsum_1/Einsum?
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_1_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp?
Itransformer_block_1/multi_head_attention_1/attention_output/einsum/EinsumEinsumCtransformer_block_1/multi_head_attention_1/einsum_1/Einsum:output:0`transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????- *
equationabcd,cde->abe2K
Itransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum?
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_1_multi_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp?
?transformer_block_1/multi_head_attention_1/attention_output/addAddV2Rtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum:output:0Vtransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2A
?transformer_block_1/multi_head_attention_1/attention_output/add?
&transformer_block_1/dropout_2/IdentityIdentityCtransformer_block_1/multi_head_attention_1/attention_output/add:z:0*
T0*+
_output_shapes
:?????????- 2(
&transformer_block_1/dropout_2/Identity?
transformer_block_1/addAddV2"average_pooling1d/Squeeze:output:0/transformer_block_1/dropout_2/Identity:output:0*
T0*+
_output_shapes
:?????????- 2
transformer_block_1/add?
Htransformer_block_1/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_1/layer_normalization_2/moments/mean/reduction_indices?
6transformer_block_1/layer_normalization_2/moments/meanMeantransformer_block_1/add:z:0Qtransformer_block_1/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(28
6transformer_block_1/layer_normalization_2/moments/mean?
>transformer_block_1/layer_normalization_2/moments/StopGradientStopGradient?transformer_block_1/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:?????????-2@
>transformer_block_1/layer_normalization_2/moments/StopGradient?
Ctransformer_block_1/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencetransformer_block_1/add:z:0Gtransformer_block_1/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????- 2E
Ctransformer_block_1/layer_normalization_2/moments/SquaredDifference?
Ltransformer_block_1/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_1/layer_normalization_2/moments/variance/reduction_indices?
:transformer_block_1/layer_normalization_2/moments/varianceMeanGtransformer_block_1/layer_normalization_2/moments/SquaredDifference:z:0Utransformer_block_1/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2<
:transformer_block_1/layer_normalization_2/moments/variance?
9transformer_block_1/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52;
9transformer_block_1/layer_normalization_2/batchnorm/add/y?
7transformer_block_1/layer_normalization_2/batchnorm/addAddV2Ctransformer_block_1/layer_normalization_2/moments/variance:output:0Btransformer_block_1/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????-29
7transformer_block_1/layer_normalization_2/batchnorm/add?
9transformer_block_1/layer_normalization_2/batchnorm/RsqrtRsqrt;transformer_block_1/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????-2;
9transformer_block_1/layer_normalization_2/batchnorm/Rsqrt?
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp?
7transformer_block_1/layer_normalization_2/batchnorm/mulMul=transformer_block_1/layer_normalization_2/batchnorm/Rsqrt:y:0Ntransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 29
7transformer_block_1/layer_normalization_2/batchnorm/mul?
9transformer_block_1/layer_normalization_2/batchnorm/mul_1Multransformer_block_1/add:z:0;transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2;
9transformer_block_1/layer_normalization_2/batchnorm/mul_1?
9transformer_block_1/layer_normalization_2/batchnorm/mul_2Mul?transformer_block_1/layer_normalization_2/moments/mean:output:0;transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2;
9transformer_block_1/layer_normalization_2/batchnorm/mul_2?
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_1_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp?
7transformer_block_1/layer_normalization_2/batchnorm/subSubJtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp:value:0=transformer_block_1/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????- 29
7transformer_block_1/layer_normalization_2/batchnorm/sub?
9transformer_block_1/layer_normalization_2/batchnorm/add_1AddV2=transformer_block_1/layer_normalization_2/batchnorm/mul_1:z:0;transformer_block_1/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????- 2;
9transformer_block_1/layer_normalization_2/batchnorm/add_1?
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_1_sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02C
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp?
7transformer_block_1/sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_1/sequential_1/dense_2/Tensordot/axes?
7transformer_block_1/sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_1/sequential_1/dense_2/Tensordot/free?
8transformer_block_1/sequential_1/dense_2/Tensordot/ShapeShape=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_2/Tensordot/Shape?
@transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axis?
;transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2GatherV2Atransformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2?
Btransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axis?
=transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1GatherV2Atransformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Ktransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1?
8transformer_block_1/sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_1/sequential_1/dense_2/Tensordot/Const?
7transformer_block_1/sequential_1/dense_2/Tensordot/ProdProdDtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Atransformer_block_1/sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_1/sequential_1/dense_2/Tensordot/Prod?
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_1?
9transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1ProdFtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1:output:0Ctransformer_block_1/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1?
>transformer_block_1/sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_1/sequential_1/dense_2/Tensordot/concat/axis?
9transformer_block_1/sequential_1/dense_2/Tensordot/concatConcatV2@transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Gtransformer_block_1/sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_1/sequential_1/dense_2/Tensordot/concat?
8transformer_block_1/sequential_1/dense_2/Tensordot/stackPack@transformer_block_1/sequential_1/dense_2/Tensordot/Prod:output:0Btransformer_block_1/sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_2/Tensordot/stack?
<transformer_block_1/sequential_1/dense_2/Tensordot/transpose	Transpose=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0Btransformer_block_1/sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2>
<transformer_block_1/sequential_1/dense_2/Tensordot/transpose?
:transformer_block_1/sequential_1/dense_2/Tensordot/ReshapeReshape@transformer_block_1/sequential_1/dense_2/Tensordot/transpose:y:0Atransformer_block_1/sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block_1/sequential_1/dense_2/Tensordot/Reshape?
9transformer_block_1/sequential_1/dense_2/Tensordot/MatMulMatMulCtransformer_block_1/sequential_1/dense_2/Tensordot/Reshape:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2;
9transformer_block_1/sequential_1/dense_2/Tensordot/MatMul?
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_2?
@transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axis?
;transformer_block_1/sequential_1/dense_2/Tensordot/concat_1ConcatV2Dtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Ctransformer_block_1/sequential_1/dense_2/Tensordot/Const_2:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_2/Tensordot/concat_1?
2transformer_block_1/sequential_1/dense_2/TensordotReshapeCtransformer_block_1/sequential_1/dense_2/Tensordot/MatMul:product:0Dtransformer_block_1/sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 24
2transformer_block_1/sequential_1/dense_2/Tensordot?
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_1_sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp?
0transformer_block_1/sequential_1/dense_2/BiasAddBiasAdd;transformer_block_1/sequential_1/dense_2/Tensordot:output:0Gtransformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 22
0transformer_block_1/sequential_1/dense_2/BiasAdd?
-transformer_block_1/sequential_1/dense_2/ReluRelu9transformer_block_1/sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????- 2/
-transformer_block_1/sequential_1/dense_2/Relu?
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_1_sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02C
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp?
7transformer_block_1/sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_1/sequential_1/dense_3/Tensordot/axes?
7transformer_block_1/sequential_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_1/sequential_1/dense_3/Tensordot/free?
8transformer_block_1/sequential_1/dense_3/Tensordot/ShapeShape;transformer_block_1/sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_3/Tensordot/Shape?
@transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axis?
;transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2GatherV2Atransformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2?
Btransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axis?
=transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1GatherV2Atransformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Ktransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1?
8transformer_block_1/sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_1/sequential_1/dense_3/Tensordot/Const?
7transformer_block_1/sequential_1/dense_3/Tensordot/ProdProdDtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Atransformer_block_1/sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_1/sequential_1/dense_3/Tensordot/Prod?
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_1?
9transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1ProdFtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1:output:0Ctransformer_block_1/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1?
>transformer_block_1/sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_1/sequential_1/dense_3/Tensordot/concat/axis?
9transformer_block_1/sequential_1/dense_3/Tensordot/concatConcatV2@transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Gtransformer_block_1/sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_1/sequential_1/dense_3/Tensordot/concat?
8transformer_block_1/sequential_1/dense_3/Tensordot/stackPack@transformer_block_1/sequential_1/dense_3/Tensordot/Prod:output:0Btransformer_block_1/sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_3/Tensordot/stack?
<transformer_block_1/sequential_1/dense_3/Tensordot/transpose	Transpose;transformer_block_1/sequential_1/dense_2/Relu:activations:0Btransformer_block_1/sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2>
<transformer_block_1/sequential_1/dense_3/Tensordot/transpose?
:transformer_block_1/sequential_1/dense_3/Tensordot/ReshapeReshape@transformer_block_1/sequential_1/dense_3/Tensordot/transpose:y:0Atransformer_block_1/sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block_1/sequential_1/dense_3/Tensordot/Reshape?
9transformer_block_1/sequential_1/dense_3/Tensordot/MatMulMatMulCtransformer_block_1/sequential_1/dense_3/Tensordot/Reshape:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2;
9transformer_block_1/sequential_1/dense_3/Tensordot/MatMul?
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_2?
@transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axis?
;transformer_block_1/sequential_1/dense_3/Tensordot/concat_1ConcatV2Dtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Ctransformer_block_1/sequential_1/dense_3/Tensordot/Const_2:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_3/Tensordot/concat_1?
2transformer_block_1/sequential_1/dense_3/TensordotReshapeCtransformer_block_1/sequential_1/dense_3/Tensordot/MatMul:product:0Dtransformer_block_1/sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 24
2transformer_block_1/sequential_1/dense_3/Tensordot?
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_1_sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp?
0transformer_block_1/sequential_1/dense_3/BiasAddBiasAdd;transformer_block_1/sequential_1/dense_3/Tensordot:output:0Gtransformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 22
0transformer_block_1/sequential_1/dense_3/BiasAdd?
&transformer_block_1/dropout_3/IdentityIdentity9transformer_block_1/sequential_1/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????- 2(
&transformer_block_1/dropout_3/Identity?
transformer_block_1/add_1AddV2=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0/transformer_block_1/dropout_3/Identity:output:0*
T0*+
_output_shapes
:?????????- 2
transformer_block_1/add_1?
Htransformer_block_1/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_1/layer_normalization_3/moments/mean/reduction_indices?
6transformer_block_1/layer_normalization_3/moments/meanMeantransformer_block_1/add_1:z:0Qtransformer_block_1/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(28
6transformer_block_1/layer_normalization_3/moments/mean?
>transformer_block_1/layer_normalization_3/moments/StopGradientStopGradient?transformer_block_1/layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:?????????-2@
>transformer_block_1/layer_normalization_3/moments/StopGradient?
Ctransformer_block_1/layer_normalization_3/moments/SquaredDifferenceSquaredDifferencetransformer_block_1/add_1:z:0Gtransformer_block_1/layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????- 2E
Ctransformer_block_1/layer_normalization_3/moments/SquaredDifference?
Ltransformer_block_1/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_1/layer_normalization_3/moments/variance/reduction_indices?
:transformer_block_1/layer_normalization_3/moments/varianceMeanGtransformer_block_1/layer_normalization_3/moments/SquaredDifference:z:0Utransformer_block_1/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2<
:transformer_block_1/layer_normalization_3/moments/variance?
9transformer_block_1/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52;
9transformer_block_1/layer_normalization_3/batchnorm/add/y?
7transformer_block_1/layer_normalization_3/batchnorm/addAddV2Ctransformer_block_1/layer_normalization_3/moments/variance:output:0Btransformer_block_1/layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????-29
7transformer_block_1/layer_normalization_3/batchnorm/add?
9transformer_block_1/layer_normalization_3/batchnorm/RsqrtRsqrt;transformer_block_1/layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????-2;
9transformer_block_1/layer_normalization_3/batchnorm/Rsqrt?
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp?
7transformer_block_1/layer_normalization_3/batchnorm/mulMul=transformer_block_1/layer_normalization_3/batchnorm/Rsqrt:y:0Ntransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 29
7transformer_block_1/layer_normalization_3/batchnorm/mul?
9transformer_block_1/layer_normalization_3/batchnorm/mul_1Multransformer_block_1/add_1:z:0;transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2;
9transformer_block_1/layer_normalization_3/batchnorm/mul_1?
9transformer_block_1/layer_normalization_3/batchnorm/mul_2Mul?transformer_block_1/layer_normalization_3/moments/mean:output:0;transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2;
9transformer_block_1/layer_normalization_3/batchnorm/mul_2?
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp?
7transformer_block_1/layer_normalization_3/batchnorm/subSubJtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp:value:0=transformer_block_1/layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????- 29
7transformer_block_1/layer_normalization_3/batchnorm/sub?
9transformer_block_1/layer_normalization_3/batchnorm/add_1AddV2=transformer_block_1/layer_normalization_3/batchnorm/mul_1:z:0;transformer_block_1/layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????- 2;
9transformer_block_1/layer_normalization_3/batchnorm/add_1?
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices?
global_average_pooling1d/MeanMean=transformer_block_1/layer_normalization_3/batchnorm/add_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2
global_average_pooling1d/Meant
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2&global_average_pooling1d/Mean:output:0inputs_1 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????(2
concatenate/concat?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:( *
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulconcatenate/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_4/Relu?
dropout_4/IdentityIdentitydense_4/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
dropout_4/Identity?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldropout_4/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_5/Relu?
dropout_5/IdentityIdentitydense_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
dropout_5/Identity?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMuldropout_5/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAdd?
IdentityIdentitydense_6/BiasAdd:output:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp8^token_and_position_embedding/embedding/embedding_lookup:^token_and_position_embedding/embedding_1/embedding_lookupC^transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpG^transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpC^transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpG^transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpO^transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpY^transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_1/multi_head_attention_1/key/add/ReadVariableOpL^transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpD^transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpN^transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpD^transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpN^transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp@^transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpB^transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp@^transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpB^transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????i:?????????::::::::::::::::::::::::2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2r
7token_and_position_embedding/embedding/embedding_lookup7token_and_position_embedding/embedding/embedding_lookup2v
9token_and_position_embedding/embedding_1/embedding_lookup9token_and_position_embedding/embedding_1/embedding_lookup2?
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpBtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp2?
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpFtransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp2?
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpBtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp2?
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpFtransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp2?
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpNtransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp2?
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2?
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOpAtransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp2?
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpKtransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2?
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOpCtransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp2?
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpMtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2?
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOpCtransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp2?
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpMtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2?
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp2?
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpAtransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp2?
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp2?
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpAtransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:R N
(
_output_shapes
:??????????i
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
#__inference_signature_wrapper_89447
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_882872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????i:?????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????i
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?5
?
@__inference_model_layer_call_and_return_conditional_losses_89334

inputs
inputs_1&
"token_and_position_embedding_89275&
"token_and_position_embedding_89277
transformer_block_1_89281
transformer_block_1_89283
transformer_block_1_89285
transformer_block_1_89287
transformer_block_1_89289
transformer_block_1_89291
transformer_block_1_89293
transformer_block_1_89295
transformer_block_1_89297
transformer_block_1_89299
transformer_block_1_89301
transformer_block_1_89303
transformer_block_1_89305
transformer_block_1_89307
transformer_block_1_89309
transformer_block_1_89311
dense_4_89316
dense_4_89318
dense_5_89322
dense_5_89324
dense_6_89328
dense_6_89330
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?4token_and_position_embedding/StatefulPartitionedCall?+transformer_block_1/StatefulPartitionedCall?
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs"token_and_position_embedding_89275"token_and_position_embedding_89277*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????i *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_8851726
4token_and_position_embedding/StatefulPartitionedCall?
!average_pooling1d/PartitionedCallPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_882962#
!average_pooling1d/PartitionedCall?
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0transformer_block_1_89281transformer_block_1_89283transformer_block_1_89285transformer_block_1_89287transformer_block_1_89289transformer_block_1_89291transformer_block_1_89293transformer_block_1_89295transformer_block_1_89297transformer_block_1_89299transformer_block_1_89301transformer_block_1_89303transformer_block_1_89305transformer_block_1_89307transformer_block_1_89309transformer_block_1_89311*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_888092-
+transformer_block_1/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall4transformer_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_889232*
(global_average_pooling1d/PartitionedCall?
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_889372
concatenate/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_89316dense_4_89318*
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
B__inference_dense_4_layer_call_and_return_conditional_losses_889572!
dense_4/StatefulPartitionedCall?
dropout_4/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_889902
dropout_4/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_5_89322dense_5_89324*
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
B__inference_dense_5_layer_call_and_return_conditional_losses_890142!
dense_5/StatefulPartitionedCall?
dropout_5/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_890472
dropout_5/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_6_89328dense_6_89330*
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
B__inference_dense_6_layer_call_and_return_conditional_losses_890702!
dense_6/StatefulPartitionedCall?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????i:?????????::::::::::::::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:P L
(
_output_shapes
:??????????i
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_89385
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_893342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????i:?????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????i
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?	
?
B__inference_dense_6_layer_call_and_return_conditional_losses_89070

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
b
)__inference_dropout_4_layer_call_fn_90405

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_889852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_88985

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*

seed*2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_89660
inputs_0
inputs_1C
?token_and_position_embedding_embedding_1_embedding_lookup_89459A
=token_and_position_embedding_embedding_embedding_lookup_89465Z
Vtransformer_block_1_multi_head_attention_1_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_1_multi_head_attention_1_query_add_readvariableop_resourceX
Ttransformer_block_1_multi_head_attention_1_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_1_multi_head_attention_1_key_add_readvariableop_resourceZ
Vtransformer_block_1_multi_head_attention_1_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_1_multi_head_attention_1_value_add_readvariableop_resourcee
atransformer_block_1_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_1_multi_head_attention_1_attention_output_add_readvariableop_resourceS
Otransformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_1_layer_normalization_2_batchnorm_readvariableop_resourceN
Jtransformer_block_1_sequential_1_dense_2_tensordot_readvariableop_resourceL
Htransformer_block_1_sequential_1_dense_2_biasadd_readvariableop_resourceN
Jtransformer_block_1_sequential_1_dense_3_tensordot_readvariableop_resourceL
Htransformer_block_1_sequential_1_dense_3_biasadd_readvariableop_resourceS
Otransformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resourceO
Ktransformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?7token_and_position_embedding/embedding/embedding_lookup?9token_and_position_embedding/embedding_1/embedding_lookup?Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp?Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp?Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp?Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp?Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp?Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp?Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp?Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp?Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp?Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp?Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp??transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp?Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp??transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp?Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp?
"token_and_position_embedding/ShapeShapeinputs_0*
T0*
_output_shapes
:2$
"token_and_position_embedding/Shape?
0token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????22
0token_and_position_embedding/strided_slice/stack?
2token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 24
2token_and_position_embedding/strided_slice/stack_1?
2token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2token_and_position_embedding/strided_slice/stack_2?
*token_and_position_embedding/strided_sliceStridedSlice+token_and_position_embedding/Shape:output:09token_and_position_embedding/strided_slice/stack:output:0;token_and_position_embedding/strided_slice/stack_1:output:0;token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*token_and_position_embedding/strided_slice?
(token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2*
(token_and_position_embedding/range/start?
(token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2*
(token_and_position_embedding/range/delta?
"token_and_position_embedding/rangeRange1token_and_position_embedding/range/start:output:03token_and_position_embedding/strided_slice:output:01token_and_position_embedding/range/delta:output:0*#
_output_shapes
:?????????2$
"token_and_position_embedding/range?
9token_and_position_embedding/embedding_1/embedding_lookupResourceGather?token_and_position_embedding_embedding_1_embedding_lookup_89459+token_and_position_embedding/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*R
_classH
FDloc:@token_and_position_embedding/embedding_1/embedding_lookup/89459*'
_output_shapes
:????????? *
dtype02;
9token_and_position_embedding/embedding_1/embedding_lookup?
Btoken_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityBtoken_and_position_embedding/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*R
_classH
FDloc:@token_and_position_embedding/embedding_1/embedding_lookup/89459*'
_output_shapes
:????????? 2D
Btoken_and_position_embedding/embedding_1/embedding_lookup/Identity?
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityKtoken_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2F
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1?
+token_and_position_embedding/embedding/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:??????????i2-
+token_and_position_embedding/embedding/Cast?
7token_and_position_embedding/embedding/embedding_lookupResourceGather=token_and_position_embedding_embedding_embedding_lookup_89465/token_and_position_embedding/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@token_and_position_embedding/embedding/embedding_lookup/89465*,
_output_shapes
:??????????i *
dtype029
7token_and_position_embedding/embedding/embedding_lookup?
@token_and_position_embedding/embedding/embedding_lookup/IdentityIdentity@token_and_position_embedding/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@token_and_position_embedding/embedding/embedding_lookup/89465*,
_output_shapes
:??????????i 2B
@token_and_position_embedding/embedding/embedding_lookup/Identity?
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityItoken_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????i 2D
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1?
 token_and_position_embedding/addAddV2Ktoken_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Mtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????i 2"
 token_and_position_embedding/add?
 average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 average_pooling1d/ExpandDims/dim?
average_pooling1d/ExpandDims
ExpandDims$token_and_position_embedding/add:z:0)average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????i 2
average_pooling1d/ExpandDims?
average_pooling1d/AvgPoolAvgPool%average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:?????????- *
ksize	
?*
paddingVALID*
strides	
?2
average_pooling1d/AvgPool?
average_pooling1d/SqueezeSqueeze"average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:?????????- *
squeeze_dims
2
average_pooling1d/Squeeze?
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_1_multi_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?
>transformer_block_1/multi_head_attention_1/query/einsum/EinsumEinsum"average_pooling1d/Squeeze:output:0Utransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2@
>transformer_block_1/multi_head_attention_1/query/einsum/Einsum?
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOpReadVariableOpLtransformer_block_1_multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp?
4transformer_block_1/multi_head_attention_1/query/addAddV2Gtransformer_block_1/multi_head_attention_1/query/einsum/Einsum:output:0Ktransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 26
4transformer_block_1/multi_head_attention_1/query/add?
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_1_multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp?
<transformer_block_1/multi_head_attention_1/key/einsum/EinsumEinsum"average_pooling1d/Squeeze:output:0Stransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2>
<transformer_block_1/multi_head_attention_1/key/einsum/Einsum?
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOpReadVariableOpJtransformer_block_1_multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp?
2transformer_block_1/multi_head_attention_1/key/addAddV2Etransformer_block_1/multi_head_attention_1/key/einsum/Einsum:output:0Itransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 24
2transformer_block_1/multi_head_attention_1/key/add?
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_1_multi_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp?
>transformer_block_1/multi_head_attention_1/value/einsum/EinsumEinsum"average_pooling1d/Squeeze:output:0Utransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2@
>transformer_block_1/multi_head_attention_1/value/einsum/Einsum?
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOpReadVariableOpLtransformer_block_1_multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp?
4transformer_block_1/multi_head_attention_1/value/addAddV2Gtransformer_block_1/multi_head_attention_1/value/einsum/Einsum:output:0Ktransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 26
4transformer_block_1/multi_head_attention_1/value/add?
0transformer_block_1/multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>22
0transformer_block_1/multi_head_attention_1/Mul/y?
.transformer_block_1/multi_head_attention_1/MulMul8transformer_block_1/multi_head_attention_1/query/add:z:09transformer_block_1/multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:?????????- 20
.transformer_block_1/multi_head_attention_1/Mul?
8transformer_block_1/multi_head_attention_1/einsum/EinsumEinsum6transformer_block_1/multi_head_attention_1/key/add:z:02transformer_block_1/multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:?????????--*
equationaecd,abcd->acbe2:
8transformer_block_1/multi_head_attention_1/einsum/Einsum?
:transformer_block_1/multi_head_attention_1/softmax/SoftmaxSoftmaxAtransformer_block_1/multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????--2<
:transformer_block_1/multi_head_attention_1/softmax/Softmax?
@transformer_block_1/multi_head_attention_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2B
@transformer_block_1/multi_head_attention_1/dropout/dropout/Const?
>transformer_block_1/multi_head_attention_1/dropout/dropout/MulMulDtransformer_block_1/multi_head_attention_1/softmax/Softmax:softmax:0Itransformer_block_1/multi_head_attention_1/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????--2@
>transformer_block_1/multi_head_attention_1/dropout/dropout/Mul?
@transformer_block_1/multi_head_attention_1/dropout/dropout/ShapeShapeDtransformer_block_1/multi_head_attention_1/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2B
@transformer_block_1/multi_head_attention_1/dropout/dropout/Shape?
Wtransformer_block_1/multi_head_attention_1/dropout/dropout/random_uniform/RandomUniformRandomUniformItransformer_block_1/multi_head_attention_1/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????--*
dtype0*

seed*2Y
Wtransformer_block_1/multi_head_attention_1/dropout/dropout/random_uniform/RandomUniform?
Itransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Itransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqual/y?
Gtransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqualGreaterEqual`transformer_block_1/multi_head_attention_1/dropout/dropout/random_uniform/RandomUniform:output:0Rtransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????--2I
Gtransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqual?
?transformer_block_1/multi_head_attention_1/dropout/dropout/CastCastKtransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????--2A
?transformer_block_1/multi_head_attention_1/dropout/dropout/Cast?
@transformer_block_1/multi_head_attention_1/dropout/dropout/Mul_1MulBtransformer_block_1/multi_head_attention_1/dropout/dropout/Mul:z:0Ctransformer_block_1/multi_head_attention_1/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????--2B
@transformer_block_1/multi_head_attention_1/dropout/dropout/Mul_1?
:transformer_block_1/multi_head_attention_1/einsum_1/EinsumEinsumDtransformer_block_1/multi_head_attention_1/dropout/dropout/Mul_1:z:08transformer_block_1/multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:?????????- *
equationacbe,aecd->abcd2<
:transformer_block_1/multi_head_attention_1/einsum_1/Einsum?
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_1_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp?
Itransformer_block_1/multi_head_attention_1/attention_output/einsum/EinsumEinsumCtransformer_block_1/multi_head_attention_1/einsum_1/Einsum:output:0`transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????- *
equationabcd,cde->abe2K
Itransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum?
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_1_multi_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp?
?transformer_block_1/multi_head_attention_1/attention_output/addAddV2Rtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum:output:0Vtransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2A
?transformer_block_1/multi_head_attention_1/attention_output/add?
+transformer_block_1/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2-
+transformer_block_1/dropout_2/dropout/Const?
)transformer_block_1/dropout_2/dropout/MulMulCtransformer_block_1/multi_head_attention_1/attention_output/add:z:04transformer_block_1/dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:?????????- 2+
)transformer_block_1/dropout_2/dropout/Mul?
+transformer_block_1/dropout_2/dropout/ShapeShapeCtransformer_block_1/multi_head_attention_1/attention_output/add:z:0*
T0*
_output_shapes
:2-
+transformer_block_1/dropout_2/dropout/Shape?
Btransformer_block_1/dropout_2/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_1/dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????- *
dtype0*

seed**
seed22D
Btransformer_block_1/dropout_2/dropout/random_uniform/RandomUniform?
4transformer_block_1/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=26
4transformer_block_1/dropout_2/dropout/GreaterEqual/y?
2transformer_block_1/dropout_2/dropout/GreaterEqualGreaterEqualKtransformer_block_1/dropout_2/dropout/random_uniform/RandomUniform:output:0=transformer_block_1/dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????- 24
2transformer_block_1/dropout_2/dropout/GreaterEqual?
*transformer_block_1/dropout_2/dropout/CastCast6transformer_block_1/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????- 2,
*transformer_block_1/dropout_2/dropout/Cast?
+transformer_block_1/dropout_2/dropout/Mul_1Mul-transformer_block_1/dropout_2/dropout/Mul:z:0.transformer_block_1/dropout_2/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????- 2-
+transformer_block_1/dropout_2/dropout/Mul_1?
transformer_block_1/addAddV2"average_pooling1d/Squeeze:output:0/transformer_block_1/dropout_2/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????- 2
transformer_block_1/add?
Htransformer_block_1/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_1/layer_normalization_2/moments/mean/reduction_indices?
6transformer_block_1/layer_normalization_2/moments/meanMeantransformer_block_1/add:z:0Qtransformer_block_1/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(28
6transformer_block_1/layer_normalization_2/moments/mean?
>transformer_block_1/layer_normalization_2/moments/StopGradientStopGradient?transformer_block_1/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:?????????-2@
>transformer_block_1/layer_normalization_2/moments/StopGradient?
Ctransformer_block_1/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencetransformer_block_1/add:z:0Gtransformer_block_1/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????- 2E
Ctransformer_block_1/layer_normalization_2/moments/SquaredDifference?
Ltransformer_block_1/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_1/layer_normalization_2/moments/variance/reduction_indices?
:transformer_block_1/layer_normalization_2/moments/varianceMeanGtransformer_block_1/layer_normalization_2/moments/SquaredDifference:z:0Utransformer_block_1/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2<
:transformer_block_1/layer_normalization_2/moments/variance?
9transformer_block_1/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52;
9transformer_block_1/layer_normalization_2/batchnorm/add/y?
7transformer_block_1/layer_normalization_2/batchnorm/addAddV2Ctransformer_block_1/layer_normalization_2/moments/variance:output:0Btransformer_block_1/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????-29
7transformer_block_1/layer_normalization_2/batchnorm/add?
9transformer_block_1/layer_normalization_2/batchnorm/RsqrtRsqrt;transformer_block_1/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????-2;
9transformer_block_1/layer_normalization_2/batchnorm/Rsqrt?
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp?
7transformer_block_1/layer_normalization_2/batchnorm/mulMul=transformer_block_1/layer_normalization_2/batchnorm/Rsqrt:y:0Ntransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 29
7transformer_block_1/layer_normalization_2/batchnorm/mul?
9transformer_block_1/layer_normalization_2/batchnorm/mul_1Multransformer_block_1/add:z:0;transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2;
9transformer_block_1/layer_normalization_2/batchnorm/mul_1?
9transformer_block_1/layer_normalization_2/batchnorm/mul_2Mul?transformer_block_1/layer_normalization_2/moments/mean:output:0;transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2;
9transformer_block_1/layer_normalization_2/batchnorm/mul_2?
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_1_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp?
7transformer_block_1/layer_normalization_2/batchnorm/subSubJtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp:value:0=transformer_block_1/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????- 29
7transformer_block_1/layer_normalization_2/batchnorm/sub?
9transformer_block_1/layer_normalization_2/batchnorm/add_1AddV2=transformer_block_1/layer_normalization_2/batchnorm/mul_1:z:0;transformer_block_1/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????- 2;
9transformer_block_1/layer_normalization_2/batchnorm/add_1?
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_1_sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02C
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp?
7transformer_block_1/sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_1/sequential_1/dense_2/Tensordot/axes?
7transformer_block_1/sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_1/sequential_1/dense_2/Tensordot/free?
8transformer_block_1/sequential_1/dense_2/Tensordot/ShapeShape=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_2/Tensordot/Shape?
@transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axis?
;transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2GatherV2Atransformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2?
Btransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axis?
=transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1GatherV2Atransformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Ktransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1?
8transformer_block_1/sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_1/sequential_1/dense_2/Tensordot/Const?
7transformer_block_1/sequential_1/dense_2/Tensordot/ProdProdDtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Atransformer_block_1/sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_1/sequential_1/dense_2/Tensordot/Prod?
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_1?
9transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1ProdFtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1:output:0Ctransformer_block_1/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1?
>transformer_block_1/sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_1/sequential_1/dense_2/Tensordot/concat/axis?
9transformer_block_1/sequential_1/dense_2/Tensordot/concatConcatV2@transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Gtransformer_block_1/sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_1/sequential_1/dense_2/Tensordot/concat?
8transformer_block_1/sequential_1/dense_2/Tensordot/stackPack@transformer_block_1/sequential_1/dense_2/Tensordot/Prod:output:0Btransformer_block_1/sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_2/Tensordot/stack?
<transformer_block_1/sequential_1/dense_2/Tensordot/transpose	Transpose=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0Btransformer_block_1/sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2>
<transformer_block_1/sequential_1/dense_2/Tensordot/transpose?
:transformer_block_1/sequential_1/dense_2/Tensordot/ReshapeReshape@transformer_block_1/sequential_1/dense_2/Tensordot/transpose:y:0Atransformer_block_1/sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block_1/sequential_1/dense_2/Tensordot/Reshape?
9transformer_block_1/sequential_1/dense_2/Tensordot/MatMulMatMulCtransformer_block_1/sequential_1/dense_2/Tensordot/Reshape:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2;
9transformer_block_1/sequential_1/dense_2/Tensordot/MatMul?
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_2?
@transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axis?
;transformer_block_1/sequential_1/dense_2/Tensordot/concat_1ConcatV2Dtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Ctransformer_block_1/sequential_1/dense_2/Tensordot/Const_2:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_2/Tensordot/concat_1?
2transformer_block_1/sequential_1/dense_2/TensordotReshapeCtransformer_block_1/sequential_1/dense_2/Tensordot/MatMul:product:0Dtransformer_block_1/sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 24
2transformer_block_1/sequential_1/dense_2/Tensordot?
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_1_sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp?
0transformer_block_1/sequential_1/dense_2/BiasAddBiasAdd;transformer_block_1/sequential_1/dense_2/Tensordot:output:0Gtransformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 22
0transformer_block_1/sequential_1/dense_2/BiasAdd?
-transformer_block_1/sequential_1/dense_2/ReluRelu9transformer_block_1/sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????- 2/
-transformer_block_1/sequential_1/dense_2/Relu?
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_1_sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02C
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp?
7transformer_block_1/sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_1/sequential_1/dense_3/Tensordot/axes?
7transformer_block_1/sequential_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_1/sequential_1/dense_3/Tensordot/free?
8transformer_block_1/sequential_1/dense_3/Tensordot/ShapeShape;transformer_block_1/sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_3/Tensordot/Shape?
@transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axis?
;transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2GatherV2Atransformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2?
Btransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axis?
=transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1GatherV2Atransformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Ktransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1?
8transformer_block_1/sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_1/sequential_1/dense_3/Tensordot/Const?
7transformer_block_1/sequential_1/dense_3/Tensordot/ProdProdDtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Atransformer_block_1/sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_1/sequential_1/dense_3/Tensordot/Prod?
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_1?
9transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1ProdFtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1:output:0Ctransformer_block_1/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1?
>transformer_block_1/sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_1/sequential_1/dense_3/Tensordot/concat/axis?
9transformer_block_1/sequential_1/dense_3/Tensordot/concatConcatV2@transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Gtransformer_block_1/sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_1/sequential_1/dense_3/Tensordot/concat?
8transformer_block_1/sequential_1/dense_3/Tensordot/stackPack@transformer_block_1/sequential_1/dense_3/Tensordot/Prod:output:0Btransformer_block_1/sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_3/Tensordot/stack?
<transformer_block_1/sequential_1/dense_3/Tensordot/transpose	Transpose;transformer_block_1/sequential_1/dense_2/Relu:activations:0Btransformer_block_1/sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2>
<transformer_block_1/sequential_1/dense_3/Tensordot/transpose?
:transformer_block_1/sequential_1/dense_3/Tensordot/ReshapeReshape@transformer_block_1/sequential_1/dense_3/Tensordot/transpose:y:0Atransformer_block_1/sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2<
:transformer_block_1/sequential_1/dense_3/Tensordot/Reshape?
9transformer_block_1/sequential_1/dense_3/Tensordot/MatMulMatMulCtransformer_block_1/sequential_1/dense_3/Tensordot/Reshape:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2;
9transformer_block_1/sequential_1/dense_3/Tensordot/MatMul?
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_2?
@transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axis?
;transformer_block_1/sequential_1/dense_3/Tensordot/concat_1ConcatV2Dtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Ctransformer_block_1/sequential_1/dense_3/Tensordot/Const_2:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_3/Tensordot/concat_1?
2transformer_block_1/sequential_1/dense_3/TensordotReshapeCtransformer_block_1/sequential_1/dense_3/Tensordot/MatMul:product:0Dtransformer_block_1/sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 24
2transformer_block_1/sequential_1/dense_3/Tensordot?
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_1_sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp?
0transformer_block_1/sequential_1/dense_3/BiasAddBiasAdd;transformer_block_1/sequential_1/dense_3/Tensordot:output:0Gtransformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 22
0transformer_block_1/sequential_1/dense_3/BiasAdd?
+transformer_block_1/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2-
+transformer_block_1/dropout_3/dropout/Const?
)transformer_block_1/dropout_3/dropout/MulMul9transformer_block_1/sequential_1/dense_3/BiasAdd:output:04transformer_block_1/dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:?????????- 2+
)transformer_block_1/dropout_3/dropout/Mul?
+transformer_block_1/dropout_3/dropout/ShapeShape9transformer_block_1/sequential_1/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2-
+transformer_block_1/dropout_3/dropout/Shape?
Btransformer_block_1/dropout_3/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_1/dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????- *
dtype0*

seed**
seed22D
Btransformer_block_1/dropout_3/dropout/random_uniform/RandomUniform?
4transformer_block_1/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=26
4transformer_block_1/dropout_3/dropout/GreaterEqual/y?
2transformer_block_1/dropout_3/dropout/GreaterEqualGreaterEqualKtransformer_block_1/dropout_3/dropout/random_uniform/RandomUniform:output:0=transformer_block_1/dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????- 24
2transformer_block_1/dropout_3/dropout/GreaterEqual?
*transformer_block_1/dropout_3/dropout/CastCast6transformer_block_1/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????- 2,
*transformer_block_1/dropout_3/dropout/Cast?
+transformer_block_1/dropout_3/dropout/Mul_1Mul-transformer_block_1/dropout_3/dropout/Mul:z:0.transformer_block_1/dropout_3/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????- 2-
+transformer_block_1/dropout_3/dropout/Mul_1?
transformer_block_1/add_1AddV2=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0/transformer_block_1/dropout_3/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????- 2
transformer_block_1/add_1?
Htransformer_block_1/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_1/layer_normalization_3/moments/mean/reduction_indices?
6transformer_block_1/layer_normalization_3/moments/meanMeantransformer_block_1/add_1:z:0Qtransformer_block_1/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(28
6transformer_block_1/layer_normalization_3/moments/mean?
>transformer_block_1/layer_normalization_3/moments/StopGradientStopGradient?transformer_block_1/layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:?????????-2@
>transformer_block_1/layer_normalization_3/moments/StopGradient?
Ctransformer_block_1/layer_normalization_3/moments/SquaredDifferenceSquaredDifferencetransformer_block_1/add_1:z:0Gtransformer_block_1/layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????- 2E
Ctransformer_block_1/layer_normalization_3/moments/SquaredDifference?
Ltransformer_block_1/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_1/layer_normalization_3/moments/variance/reduction_indices?
:transformer_block_1/layer_normalization_3/moments/varianceMeanGtransformer_block_1/layer_normalization_3/moments/SquaredDifference:z:0Utransformer_block_1/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2<
:transformer_block_1/layer_normalization_3/moments/variance?
9transformer_block_1/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52;
9transformer_block_1/layer_normalization_3/batchnorm/add/y?
7transformer_block_1/layer_normalization_3/batchnorm/addAddV2Ctransformer_block_1/layer_normalization_3/moments/variance:output:0Btransformer_block_1/layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????-29
7transformer_block_1/layer_normalization_3/batchnorm/add?
9transformer_block_1/layer_normalization_3/batchnorm/RsqrtRsqrt;transformer_block_1/layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????-2;
9transformer_block_1/layer_normalization_3/batchnorm/Rsqrt?
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp?
7transformer_block_1/layer_normalization_3/batchnorm/mulMul=transformer_block_1/layer_normalization_3/batchnorm/Rsqrt:y:0Ntransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 29
7transformer_block_1/layer_normalization_3/batchnorm/mul?
9transformer_block_1/layer_normalization_3/batchnorm/mul_1Multransformer_block_1/add_1:z:0;transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2;
9transformer_block_1/layer_normalization_3/batchnorm/mul_1?
9transformer_block_1/layer_normalization_3/batchnorm/mul_2Mul?transformer_block_1/layer_normalization_3/moments/mean:output:0;transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2;
9transformer_block_1/layer_normalization_3/batchnorm/mul_2?
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp?
7transformer_block_1/layer_normalization_3/batchnorm/subSubJtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp:value:0=transformer_block_1/layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????- 29
7transformer_block_1/layer_normalization_3/batchnorm/sub?
9transformer_block_1/layer_normalization_3/batchnorm/add_1AddV2=transformer_block_1/layer_normalization_3/batchnorm/mul_1:z:0;transformer_block_1/layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????- 2;
9transformer_block_1/layer_normalization_3/batchnorm/add_1?
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indices?
global_average_pooling1d/MeanMean=transformer_block_1/layer_normalization_3/batchnorm/add_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2
global_average_pooling1d/Meant
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2&global_average_pooling1d/Mean:output:0inputs_1 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????(2
concatenate/concat?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:( *
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulconcatenate/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_4/Reluw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_4/dropout/Const?
dropout_4/dropout/MulMuldense_4/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_4/dropout/Mul|
dropout_4/dropout/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*

seed**
seed220
.dropout_4/dropout/random_uniform/RandomUniform?
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_4/dropout/GreaterEqual/y?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_4/dropout/Mul_1?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldropout_4/dropout/Mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_5/Reluw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_5/dropout/Const?
dropout_5/dropout/MulMuldense_5/Relu:activations:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_5/dropout/Mul|
dropout_5/dropout/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape?
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*

seed**
seed220
.dropout_5/dropout/random_uniform/RandomUniform?
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_5/dropout/GreaterEqual/y?
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2 
dropout_5/dropout/GreaterEqual?
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_5/dropout/Cast?
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_5/dropout/Mul_1?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMuldropout_5/dropout/Mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAdd?
IdentityIdentitydense_6/BiasAdd:output:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp8^token_and_position_embedding/embedding/embedding_lookup:^token_and_position_embedding/embedding_1/embedding_lookupC^transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpG^transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpC^transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpG^transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpO^transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpY^transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_1/multi_head_attention_1/key/add/ReadVariableOpL^transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpD^transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpN^transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpD^transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpN^transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp@^transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpB^transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp@^transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpB^transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????i:?????????::::::::::::::::::::::::2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2r
7token_and_position_embedding/embedding/embedding_lookup7token_and_position_embedding/embedding/embedding_lookup2v
9token_and_position_embedding/embedding_1/embedding_lookup9token_and_position_embedding/embedding_1/embedding_lookup2?
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpBtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp2?
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpFtransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp2?
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpBtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp2?
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpFtransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp2?
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpNtransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp2?
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2?
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOpAtransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp2?
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpKtransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2?
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOpCtransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp2?
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpMtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2?
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOpCtransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp2?
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpMtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2?
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp2?
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpAtransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp2?
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp2?
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpAtransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:R N
(
_output_shapes
:??????????i
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?H
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_90533

inputs-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource-
)dense_3_tensordot_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity??dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp? dense_3/Tensordot/ReadVariableOp?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes?
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/freeh
dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape?
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2?
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod?
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1?
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack?
dense_2/Tensordot/transpose	Transposeinputs!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2
dense_2/Tensordot/transpose?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_2?
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2
dense_2/BiasAddt
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????- 2
dense_2/Relu?
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02"
 dense_3/Tensordot/ReadVariableOpz
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/axes?
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_3/Tensordot/free|
dense_3/Tensordot/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2
dense_3/Tensordot/Shape?
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/GatherV2/axis?
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2?
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axis?
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2_1|
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const?
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod?
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1?
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1?
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axis?
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat?
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stack?
dense_3/Tensordot/transpose	Transposedense_2/Relu:activations:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2
dense_3/Tensordot/transpose?
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_3/Tensordot/Reshape?
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_3/Tensordot/MatMul?
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_2?
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axis?
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1?
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2
dense_3/Tensordot?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2
dense_3/BiasAdd?
IdentityIdentitydense_3/BiasAdd:output:0^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????- ::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
?
p
F__inference_concatenate_layer_call_and_return_conditional_losses_88937

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????(2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :?????????:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_89970
x&
"embedding_1_embedding_lookup_89957$
 embedding_embedding_lookup_89963
identity??embedding/embedding_lookup?embedding_1/embedding_lookup?
ShapeShapex*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/delta?
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:?????????2
range?
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_89957range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/89957*'
_output_shapes
:????????? *
dtype02
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/89957*'
_output_shapes
:????????? 2'
%embedding_1/embedding_lookup/Identity?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2)
'embedding_1/embedding_lookup/Identity_1m
embedding/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:??????????i2
embedding/Cast?
embedding/embedding_lookupResourceGather embedding_embedding_lookup_89963embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/89963*,
_output_shapes
:??????????i *
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/89963*,
_output_shapes
:??????????i 2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????i 2'
%embedding/embedding_lookup/Identity_1?
addAddV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????i 2
add?
IdentityIdentityadd:z:0^embedding/embedding_lookup^embedding_1/embedding_lookup*
T0*,
_output_shapes
:??????????i 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????i::28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:K G
(
_output_shapes
:??????????i

_user_specified_namex
??
?
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_90254

inputsF
Bmulti_head_attention_1_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_1_query_add_readvariableop_resourceD
@multi_head_attention_1_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_1_key_add_readvariableop_resourceF
Bmulti_head_attention_1_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_1_value_add_readvariableop_resourceQ
Mmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_1_attention_output_add_readvariableop_resource?
;layer_normalization_2_batchnorm_mul_readvariableop_resource;
7layer_normalization_2_batchnorm_readvariableop_resource:
6sequential_1_dense_2_tensordot_readvariableop_resource8
4sequential_1_dense_2_biasadd_readvariableop_resource:
6sequential_1_dense_3_tensordot_readvariableop_resource8
4sequential_1_dense_3_biasadd_readvariableop_resource?
;layer_normalization_3_batchnorm_mul_readvariableop_resource;
7layer_normalization_3_batchnorm_readvariableop_resource
identity??.layer_normalization_2/batchnorm/ReadVariableOp?2layer_normalization_2/batchnorm/mul/ReadVariableOp?.layer_normalization_3/batchnorm/ReadVariableOp?2layer_normalization_3/batchnorm/mul/ReadVariableOp?:multi_head_attention_1/attention_output/add/ReadVariableOp?Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp?-multi_head_attention_1/key/add/ReadVariableOp?7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp?/multi_head_attention_1/query/add/ReadVariableOp?9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?/multi_head_attention_1/value/add/ReadVariableOp?9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp?+sequential_1/dense_2/BiasAdd/ReadVariableOp?-sequential_1/dense_2/Tensordot/ReadVariableOp?+sequential_1/dense_3/BiasAdd/ReadVariableOp?-sequential_1/dense_3/Tensordot/ReadVariableOp?
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?
*multi_head_attention_1/query/einsum/EinsumEinsuminputsAmulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/Einsum?
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/query/add/ReadVariableOp?
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 2"
 multi_head_attention_1/query/add?
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp?
(multi_head_attention_1/key/einsum/EinsumEinsuminputs?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/Einsum?
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_1/key/add/ReadVariableOp?
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 2 
multi_head_attention_1/key/add?
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp?
*multi_head_attention_1/value/einsum/EinsumEinsuminputsAmulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/Einsum?
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/value/add/ReadVariableOp?
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 2"
 multi_head_attention_1/value/add?
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>2
multi_head_attention_1/Mul/y?
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:?????????- 2
multi_head_attention_1/Mul?
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:?????????--*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/Einsum?
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????--2(
&multi_head_attention_1/softmax/Softmax?
'multi_head_attention_1/dropout/IdentityIdentity0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????--2)
'multi_head_attention_1/dropout/Identity?
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/Identity:output:0$multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:?????????- *
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/Einsum?
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp?
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????- *
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/Einsum?
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOp?
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2-
+multi_head_attention_1/attention_output/add?
dropout_2/IdentityIdentity/multi_head_attention_1/attention_output/add:z:0*
T0*+
_output_shapes
:?????????- 2
dropout_2/Identityn
addAddV2inputsdropout_2/Identity:output:0*
T0*+
_output_shapes
:?????????- 2
add?
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indices?
"layer_normalization_2/moments/meanMeanadd:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2$
"layer_normalization_2/moments/mean?
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:?????????-2,
*layer_normalization_2/moments/StopGradient?
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????- 21
/layer_normalization_2/moments/SquaredDifference?
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indices?
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2(
&layer_normalization_2/moments/variance?
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_2/batchnorm/add/y?
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????-2%
#layer_normalization_2/batchnorm/add?
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????-2'
%layer_normalization_2/batchnorm/Rsqrt?
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOp?
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2%
#layer_normalization_2/batchnorm/mul?
%layer_normalization_2/batchnorm/mul_1Muladd:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_2/batchnorm/mul_1?
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_2/batchnorm/mul_2?
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_2/batchnorm/ReadVariableOp?
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????- 2%
#layer_normalization_2/batchnorm/sub?
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_2/batchnorm/add_1?
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_1/dense_2/Tensordot/ReadVariableOp?
#sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_2/Tensordot/axes?
#sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_2/Tensordot/free?
$sequential_1/dense_2/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/Shape?
,sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/GatherV2/axis?
'sequential_1/dense_2/Tensordot/GatherV2GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/free:output:05sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/GatherV2?
.sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_2/Tensordot/GatherV2_1/axis?
)sequential_1/dense_2/Tensordot/GatherV2_1GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/axes:output:07sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_2/Tensordot/GatherV2_1?
$sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_2/Tensordot/Const?
#sequential_1/dense_2/Tensordot/ProdProd0sequential_1/dense_2/Tensordot/GatherV2:output:0-sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_2/Tensordot/Prod?
&sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_1?
%sequential_1/dense_2/Tensordot/Prod_1Prod2sequential_1/dense_2/Tensordot/GatherV2_1:output:0/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_2/Tensordot/Prod_1?
*sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_2/Tensordot/concat/axis?
%sequential_1/dense_2/Tensordot/concatConcatV2,sequential_1/dense_2/Tensordot/free:output:0,sequential_1/dense_2/Tensordot/axes:output:03sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_2/Tensordot/concat?
$sequential_1/dense_2/Tensordot/stackPack,sequential_1/dense_2/Tensordot/Prod:output:0.sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/stack?
(sequential_1/dense_2/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0.sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2*
(sequential_1/dense_2/Tensordot/transpose?
&sequential_1/dense_2/Tensordot/ReshapeReshape,sequential_1/dense_2/Tensordot/transpose:y:0-sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&sequential_1/dense_2/Tensordot/Reshape?
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%sequential_1/dense_2/Tensordot/MatMul?
&sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_2?
,sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/concat_1/axis?
'sequential_1/dense_2/Tensordot/concat_1ConcatV20sequential_1/dense_2/Tensordot/GatherV2:output:0/sequential_1/dense_2/Tensordot/Const_2:output:05sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/concat_1?
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:00sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2 
sequential_1/dense_2/Tensordot?
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOp?
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2
sequential_1/dense_2/BiasAdd?
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????- 2
sequential_1/dense_2/Relu?
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_1/dense_3/Tensordot/ReadVariableOp?
#sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_3/Tensordot/axes?
#sequential_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_3/Tensordot/free?
$sequential_1/dense_3/Tensordot/ShapeShape'sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/Shape?
,sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/GatherV2/axis?
'sequential_1/dense_3/Tensordot/GatherV2GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/free:output:05sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/GatherV2?
.sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_3/Tensordot/GatherV2_1/axis?
)sequential_1/dense_3/Tensordot/GatherV2_1GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/axes:output:07sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_3/Tensordot/GatherV2_1?
$sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_3/Tensordot/Const?
#sequential_1/dense_3/Tensordot/ProdProd0sequential_1/dense_3/Tensordot/GatherV2:output:0-sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_3/Tensordot/Prod?
&sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_1?
%sequential_1/dense_3/Tensordot/Prod_1Prod2sequential_1/dense_3/Tensordot/GatherV2_1:output:0/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_3/Tensordot/Prod_1?
*sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_3/Tensordot/concat/axis?
%sequential_1/dense_3/Tensordot/concatConcatV2,sequential_1/dense_3/Tensordot/free:output:0,sequential_1/dense_3/Tensordot/axes:output:03sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_3/Tensordot/concat?
$sequential_1/dense_3/Tensordot/stackPack,sequential_1/dense_3/Tensordot/Prod:output:0.sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/stack?
(sequential_1/dense_3/Tensordot/transpose	Transpose'sequential_1/dense_2/Relu:activations:0.sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2*
(sequential_1/dense_3/Tensordot/transpose?
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_3/Tensordot/transpose:y:0-sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&sequential_1/dense_3/Tensordot/Reshape?
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%sequential_1/dense_3/Tensordot/MatMul?
&sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_2?
,sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/concat_1/axis?
'sequential_1/dense_3/Tensordot/concat_1ConcatV20sequential_1/dense_3/Tensordot/GatherV2:output:0/sequential_1/dense_3/Tensordot/Const_2:output:05sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/concat_1?
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:00sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2 
sequential_1/dense_3/Tensordot?
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOp?
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2
sequential_1/dense_3/BiasAdd?
dropout_3/IdentityIdentity%sequential_1/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????- 2
dropout_3/Identity?
add_1AddV2)layer_normalization_2/batchnorm/add_1:z:0dropout_3/Identity:output:0*
T0*+
_output_shapes
:?????????- 2
add_1?
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indices?
"layer_normalization_3/moments/meanMean	add_1:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2$
"layer_normalization_3/moments/mean?
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:?????????-2,
*layer_normalization_3/moments/StopGradient?
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????- 21
/layer_normalization_3/moments/SquaredDifference?
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indices?
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2(
&layer_normalization_3/moments/variance?
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_3/batchnorm/add/y?
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????-2%
#layer_normalization_3/batchnorm/add?
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????-2'
%layer_normalization_3/batchnorm/Rsqrt?
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOp?
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2%
#layer_normalization_3/batchnorm/mul?
%layer_normalization_3/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_3/batchnorm/mul_1?
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_3/batchnorm/mul_2?
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_3/batchnorm/ReadVariableOp?
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????- 2%
#layer_normalization_3/batchnorm/sub?
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_3/batchnorm/add_1?
IdentityIdentity)layer_normalization_3/batchnorm/add_1:z:0/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:?????????- ::::::::::::::::2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2?
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_1/key/add/ReadVariableOp-multi_head_attention_1/key/add/ReadVariableOp2r
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/query/add/ReadVariableOp/multi_head_attention_1/query/add/ReadVariableOp2v
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/value/add/ReadVariableOp/multi_head_attention_1/value/add/ReadVariableOp2v
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2^
-sequential_1/dense_2/Tensordot/ReadVariableOp-sequential_1/dense_2/Tensordot/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2^
-sequential_1/dense_3/Tensordot/ReadVariableOp-sequential_1/dense_3/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
?8
?
@__inference_model_layer_call_and_return_conditional_losses_89217

inputs
inputs_1&
"token_and_position_embedding_89158&
"token_and_position_embedding_89160
transformer_block_1_89164
transformer_block_1_89166
transformer_block_1_89168
transformer_block_1_89170
transformer_block_1_89172
transformer_block_1_89174
transformer_block_1_89176
transformer_block_1_89178
transformer_block_1_89180
transformer_block_1_89182
transformer_block_1_89184
transformer_block_1_89186
transformer_block_1_89188
transformer_block_1_89190
transformer_block_1_89192
transformer_block_1_89194
dense_4_89199
dense_4_89201
dense_5_89205
dense_5_89207
dense_6_89211
dense_6_89213
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?4token_and_position_embedding/StatefulPartitionedCall?+transformer_block_1/StatefulPartitionedCall?
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs"token_and_position_embedding_89158"token_and_position_embedding_89160*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????i *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_8851726
4token_and_position_embedding/StatefulPartitionedCall?
!average_pooling1d/PartitionedCallPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_882962#
!average_pooling1d/PartitionedCall?
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0transformer_block_1_89164transformer_block_1_89166transformer_block_1_89168transformer_block_1_89170transformer_block_1_89172transformer_block_1_89174transformer_block_1_89176transformer_block_1_89178transformer_block_1_89180transformer_block_1_89182transformer_block_1_89184transformer_block_1_89186transformer_block_1_89188transformer_block_1_89190transformer_block_1_89192transformer_block_1_89194*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_886822-
+transformer_block_1/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall4transformer_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_889232*
(global_average_pooling1d/PartitionedCall?
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_889372
concatenate/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_89199dense_4_89201*
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
B__inference_dense_4_layer_call_and_return_conditional_losses_889572!
dense_4/StatefulPartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_889852#
!dropout_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_5_89205dense_5_89207*
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
B__inference_dense_5_layer_call_and_return_conditional_losses_890142!
dense_5/StatefulPartitionedCall?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_890422#
!dropout_5/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_6_89211dense_6_89213*
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
B__inference_dense_6_layer_call_and_return_conditional_losses_890702!
dense_6/StatefulPartitionedCall?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????i:?????????::::::::::::::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:P L
(
_output_shapes
:??????????i
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_90334

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
|
'__inference_dense_6_layer_call_fn_90476

inputs
unknown
	unknown_0
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
B__inference_dense_6_layer_call_and_return_conditional_losses_890702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
<__inference_token_and_position_embedding_layer_call_fn_89979
x
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????i *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_885172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????i 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????i::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????i

_user_specified_namex
??
?
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_90127

inputsF
Bmulti_head_attention_1_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_1_query_add_readvariableop_resourceD
@multi_head_attention_1_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_1_key_add_readvariableop_resourceF
Bmulti_head_attention_1_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_1_value_add_readvariableop_resourceQ
Mmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_1_attention_output_add_readvariableop_resource?
;layer_normalization_2_batchnorm_mul_readvariableop_resource;
7layer_normalization_2_batchnorm_readvariableop_resource:
6sequential_1_dense_2_tensordot_readvariableop_resource8
4sequential_1_dense_2_biasadd_readvariableop_resource:
6sequential_1_dense_3_tensordot_readvariableop_resource8
4sequential_1_dense_3_biasadd_readvariableop_resource?
;layer_normalization_3_batchnorm_mul_readvariableop_resource;
7layer_normalization_3_batchnorm_readvariableop_resource
identity??.layer_normalization_2/batchnorm/ReadVariableOp?2layer_normalization_2/batchnorm/mul/ReadVariableOp?.layer_normalization_3/batchnorm/ReadVariableOp?2layer_normalization_3/batchnorm/mul/ReadVariableOp?:multi_head_attention_1/attention_output/add/ReadVariableOp?Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp?-multi_head_attention_1/key/add/ReadVariableOp?7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp?/multi_head_attention_1/query/add/ReadVariableOp?9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?/multi_head_attention_1/value/add/ReadVariableOp?9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp?+sequential_1/dense_2/BiasAdd/ReadVariableOp?-sequential_1/dense_2/Tensordot/ReadVariableOp?+sequential_1/dense_3/BiasAdd/ReadVariableOp?-sequential_1/dense_3/Tensordot/ReadVariableOp?
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?
*multi_head_attention_1/query/einsum/EinsumEinsuminputsAmulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/Einsum?
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/query/add/ReadVariableOp?
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 2"
 multi_head_attention_1/query/add?
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp?
(multi_head_attention_1/key/einsum/EinsumEinsuminputs?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/Einsum?
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_1/key/add/ReadVariableOp?
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 2 
multi_head_attention_1/key/add?
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp?
*multi_head_attention_1/value/einsum/EinsumEinsuminputsAmulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/Einsum?
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/value/add/ReadVariableOp?
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 2"
 multi_head_attention_1/value/add?
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>2
multi_head_attention_1/Mul/y?
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:?????????- 2
multi_head_attention_1/Mul?
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:?????????--*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/Einsum?
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????--2(
&multi_head_attention_1/softmax/Softmax?
,multi_head_attention_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,multi_head_attention_1/dropout/dropout/Const?
*multi_head_attention_1/dropout/dropout/MulMul0multi_head_attention_1/softmax/Softmax:softmax:05multi_head_attention_1/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????--2,
*multi_head_attention_1/dropout/dropout/Mul?
,multi_head_attention_1/dropout/dropout/ShapeShape0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_1/dropout/dropout/Shape?
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_1/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????--*
dtype0*

seed*2E
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform?
5multi_head_attention_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_1/dropout/dropout/GreaterEqual/y?
3multi_head_attention_1/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_1/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????--25
3multi_head_attention_1/dropout/dropout/GreaterEqual?
+multi_head_attention_1/dropout/dropout/CastCast7multi_head_attention_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????--2-
+multi_head_attention_1/dropout/dropout/Cast?
,multi_head_attention_1/dropout/dropout/Mul_1Mul.multi_head_attention_1/dropout/dropout/Mul:z:0/multi_head_attention_1/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????--2.
,multi_head_attention_1/dropout/dropout/Mul_1?
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/dropout/Mul_1:z:0$multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:?????????- *
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/Einsum?
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp?
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????- *
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/Einsum?
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOp?
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2-
+multi_head_attention_1/attention_output/addw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMul/multi_head_attention_1/attention_output/add:z:0 dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:?????????- 2
dropout_2/dropout/Mul?
dropout_2/dropout/ShapeShape/multi_head_attention_1/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????- *
dtype0*

seed**
seed220
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????- 2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????- 2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????- 2
dropout_2/dropout/Mul_1n
addAddV2inputsdropout_2/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????- 2
add?
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indices?
"layer_normalization_2/moments/meanMeanadd:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2$
"layer_normalization_2/moments/mean?
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:?????????-2,
*layer_normalization_2/moments/StopGradient?
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????- 21
/layer_normalization_2/moments/SquaredDifference?
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indices?
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2(
&layer_normalization_2/moments/variance?
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_2/batchnorm/add/y?
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????-2%
#layer_normalization_2/batchnorm/add?
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????-2'
%layer_normalization_2/batchnorm/Rsqrt?
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOp?
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2%
#layer_normalization_2/batchnorm/mul?
%layer_normalization_2/batchnorm/mul_1Muladd:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_2/batchnorm/mul_1?
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_2/batchnorm/mul_2?
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_2/batchnorm/ReadVariableOp?
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????- 2%
#layer_normalization_2/batchnorm/sub?
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_2/batchnorm/add_1?
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_1/dense_2/Tensordot/ReadVariableOp?
#sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_2/Tensordot/axes?
#sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_2/Tensordot/free?
$sequential_1/dense_2/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/Shape?
,sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/GatherV2/axis?
'sequential_1/dense_2/Tensordot/GatherV2GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/free:output:05sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/GatherV2?
.sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_2/Tensordot/GatherV2_1/axis?
)sequential_1/dense_2/Tensordot/GatherV2_1GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/axes:output:07sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_2/Tensordot/GatherV2_1?
$sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_2/Tensordot/Const?
#sequential_1/dense_2/Tensordot/ProdProd0sequential_1/dense_2/Tensordot/GatherV2:output:0-sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_2/Tensordot/Prod?
&sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_1?
%sequential_1/dense_2/Tensordot/Prod_1Prod2sequential_1/dense_2/Tensordot/GatherV2_1:output:0/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_2/Tensordot/Prod_1?
*sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_2/Tensordot/concat/axis?
%sequential_1/dense_2/Tensordot/concatConcatV2,sequential_1/dense_2/Tensordot/free:output:0,sequential_1/dense_2/Tensordot/axes:output:03sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_2/Tensordot/concat?
$sequential_1/dense_2/Tensordot/stackPack,sequential_1/dense_2/Tensordot/Prod:output:0.sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/stack?
(sequential_1/dense_2/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0.sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2*
(sequential_1/dense_2/Tensordot/transpose?
&sequential_1/dense_2/Tensordot/ReshapeReshape,sequential_1/dense_2/Tensordot/transpose:y:0-sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&sequential_1/dense_2/Tensordot/Reshape?
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%sequential_1/dense_2/Tensordot/MatMul?
&sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_2?
,sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/concat_1/axis?
'sequential_1/dense_2/Tensordot/concat_1ConcatV20sequential_1/dense_2/Tensordot/GatherV2:output:0/sequential_1/dense_2/Tensordot/Const_2:output:05sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/concat_1?
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:00sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2 
sequential_1/dense_2/Tensordot?
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOp?
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2
sequential_1/dense_2/BiasAdd?
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????- 2
sequential_1/dense_2/Relu?
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_1/dense_3/Tensordot/ReadVariableOp?
#sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_3/Tensordot/axes?
#sequential_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_3/Tensordot/free?
$sequential_1/dense_3/Tensordot/ShapeShape'sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/Shape?
,sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/GatherV2/axis?
'sequential_1/dense_3/Tensordot/GatherV2GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/free:output:05sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/GatherV2?
.sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_3/Tensordot/GatherV2_1/axis?
)sequential_1/dense_3/Tensordot/GatherV2_1GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/axes:output:07sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_3/Tensordot/GatherV2_1?
$sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_3/Tensordot/Const?
#sequential_1/dense_3/Tensordot/ProdProd0sequential_1/dense_3/Tensordot/GatherV2:output:0-sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_3/Tensordot/Prod?
&sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_1?
%sequential_1/dense_3/Tensordot/Prod_1Prod2sequential_1/dense_3/Tensordot/GatherV2_1:output:0/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_3/Tensordot/Prod_1?
*sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_3/Tensordot/concat/axis?
%sequential_1/dense_3/Tensordot/concatConcatV2,sequential_1/dense_3/Tensordot/free:output:0,sequential_1/dense_3/Tensordot/axes:output:03sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_3/Tensordot/concat?
$sequential_1/dense_3/Tensordot/stackPack,sequential_1/dense_3/Tensordot/Prod:output:0.sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/stack?
(sequential_1/dense_3/Tensordot/transpose	Transpose'sequential_1/dense_2/Relu:activations:0.sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2*
(sequential_1/dense_3/Tensordot/transpose?
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_3/Tensordot/transpose:y:0-sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&sequential_1/dense_3/Tensordot/Reshape?
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%sequential_1/dense_3/Tensordot/MatMul?
&sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_2?
,sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/concat_1/axis?
'sequential_1/dense_3/Tensordot/concat_1ConcatV20sequential_1/dense_3/Tensordot/GatherV2:output:0/sequential_1/dense_3/Tensordot/Const_2:output:05sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/concat_1?
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:00sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2 
sequential_1/dense_3/Tensordot?
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOp?
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2
sequential_1/dense_3/BiasAddw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_3/dropout/Const?
dropout_3/dropout/MulMul%sequential_1/dense_3/BiasAdd:output:0 dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:?????????- 2
dropout_3/dropout/Mul?
dropout_3/dropout/ShapeShape%sequential_1/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????- *
dtype0*

seed**
seed220
.dropout_3/dropout/random_uniform/RandomUniform?
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_3/dropout/GreaterEqual/y?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????- 2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????- 2
dropout_3/dropout/Cast?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????- 2
dropout_3/dropout/Mul_1?
add_1AddV2)layer_normalization_2/batchnorm/add_1:z:0dropout_3/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????- 2
add_1?
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indices?
"layer_normalization_3/moments/meanMean	add_1:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2$
"layer_normalization_3/moments/mean?
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:?????????-2,
*layer_normalization_3/moments/StopGradient?
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????- 21
/layer_normalization_3/moments/SquaredDifference?
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indices?
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2(
&layer_normalization_3/moments/variance?
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_3/batchnorm/add/y?
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????-2%
#layer_normalization_3/batchnorm/add?
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????-2'
%layer_normalization_3/batchnorm/Rsqrt?
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOp?
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2%
#layer_normalization_3/batchnorm/mul?
%layer_normalization_3/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_3/batchnorm/mul_1?
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_3/batchnorm/mul_2?
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_3/batchnorm/ReadVariableOp?
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????- 2%
#layer_normalization_3/batchnorm/sub?
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_3/batchnorm/add_1?
IdentityIdentity)layer_normalization_3/batchnorm/add_1:z:0/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:?????????- ::::::::::::::::2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2?
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_1/key/add/ReadVariableOp-multi_head_attention_1/key/add/ReadVariableOp2r
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/query/add/ReadVariableOp/multi_head_attention_1/query/add/ReadVariableOp2v
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/value/add/ReadVariableOp/multi_head_attention_1/value/add/ReadVariableOp2v
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2^
-sequential_1/dense_2/Tensordot/ReadVariableOp-sequential_1/dense_2/Tensordot/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2^
-sequential_1/dense_3/Tensordot/ReadVariableOp-sequential_1/dense_3/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
?
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_89042

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*

seed*2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
B__inference_dense_3_layer_call_and_return_conditional_losses_90686

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????- ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_88400
dense_2_input
dense_2_88348
dense_2_88350
dense_3_88394
dense_3_88396
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCalldense_2_inputdense_2_88348dense_2_88350*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_883372!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_88394dense_3_88396*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_883832!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????- ::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????- 
'
_user_specified_namedense_2_input
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_88485

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_88517
x&
"embedding_1_embedding_lookup_88504$
 embedding_embedding_lookup_88510
identity??embedding/embedding_lookup?embedding_1/embedding_lookup?
ShapeShapex*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/delta?
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:?????????2
range?
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_88504range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/88504*'
_output_shapes
:????????? *
dtype02
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/88504*'
_output_shapes
:????????? 2'
%embedding_1/embedding_lookup/Identity?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2)
'embedding_1/embedding_lookup/Identity_1m
embedding/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:??????????i2
embedding/Cast?
embedding/embedding_lookupResourceGather embedding_embedding_lookup_88510embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/88510*,
_output_shapes
:??????????i *
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/88510*,
_output_shapes
:??????????i 2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????i 2'
%embedding/embedding_lookup/Identity_1?
addAddV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????i 2
add?
IdentityIdentityadd:z:0^embedding/embedding_lookup^embedding_1/embedding_lookup*
T0*,
_output_shapes
:??????????i 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????i::28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:K G
(
_output_shapes
:??????????i

_user_specified_namex
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_90400

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_88431

inputs
dense_2_88420
dense_2_88422
dense_3_88425
dense_3_88427
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_88420dense_2_88422*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_883372!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_88425dense_3_88427*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_883832!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????- ::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
?
|
'__inference_dense_3_layer_call_fn_90695

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_883832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????- ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_88414
dense_2_input
dense_2_88403
dense_2_88405
dense_3_88408
dense_3_88410
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCalldense_2_inputdense_2_88403dense_2_88405*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_883372!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_88408dense_3_88410*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_883832!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????- ::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????- 
'
_user_specified_namedense_2_input
?
?
%__inference_model_layer_call_fn_89268
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_892172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????i:?????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????i
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
%__inference_model_layer_call_fn_89892
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_892172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????i:?????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????i
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
??
?
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_88682

inputsF
Bmulti_head_attention_1_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_1_query_add_readvariableop_resourceD
@multi_head_attention_1_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_1_key_add_readvariableop_resourceF
Bmulti_head_attention_1_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_1_value_add_readvariableop_resourceQ
Mmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_1_attention_output_add_readvariableop_resource?
;layer_normalization_2_batchnorm_mul_readvariableop_resource;
7layer_normalization_2_batchnorm_readvariableop_resource:
6sequential_1_dense_2_tensordot_readvariableop_resource8
4sequential_1_dense_2_biasadd_readvariableop_resource:
6sequential_1_dense_3_tensordot_readvariableop_resource8
4sequential_1_dense_3_biasadd_readvariableop_resource?
;layer_normalization_3_batchnorm_mul_readvariableop_resource;
7layer_normalization_3_batchnorm_readvariableop_resource
identity??.layer_normalization_2/batchnorm/ReadVariableOp?2layer_normalization_2/batchnorm/mul/ReadVariableOp?.layer_normalization_3/batchnorm/ReadVariableOp?2layer_normalization_3/batchnorm/mul/ReadVariableOp?:multi_head_attention_1/attention_output/add/ReadVariableOp?Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp?-multi_head_attention_1/key/add/ReadVariableOp?7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp?/multi_head_attention_1/query/add/ReadVariableOp?9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?/multi_head_attention_1/value/add/ReadVariableOp?9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp?+sequential_1/dense_2/BiasAdd/ReadVariableOp?-sequential_1/dense_2/Tensordot/ReadVariableOp?+sequential_1/dense_3/BiasAdd/ReadVariableOp?-sequential_1/dense_3/Tensordot/ReadVariableOp?
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?
*multi_head_attention_1/query/einsum/EinsumEinsuminputsAmulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/Einsum?
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/query/add/ReadVariableOp?
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 2"
 multi_head_attention_1/query/add?
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp?
(multi_head_attention_1/key/einsum/EinsumEinsuminputs?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/Einsum?
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_1/key/add/ReadVariableOp?
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 2 
multi_head_attention_1/key/add?
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp?
*multi_head_attention_1/value/einsum/EinsumEinsuminputsAmulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/Einsum?
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/value/add/ReadVariableOp?
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 2"
 multi_head_attention_1/value/add?
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>2
multi_head_attention_1/Mul/y?
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:?????????- 2
multi_head_attention_1/Mul?
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:?????????--*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/Einsum?
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????--2(
&multi_head_attention_1/softmax/Softmax?
,multi_head_attention_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,multi_head_attention_1/dropout/dropout/Const?
*multi_head_attention_1/dropout/dropout/MulMul0multi_head_attention_1/softmax/Softmax:softmax:05multi_head_attention_1/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????--2,
*multi_head_attention_1/dropout/dropout/Mul?
,multi_head_attention_1/dropout/dropout/ShapeShape0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_1/dropout/dropout/Shape?
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_1/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????--*
dtype0*

seed*2E
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform?
5multi_head_attention_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_1/dropout/dropout/GreaterEqual/y?
3multi_head_attention_1/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_1/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????--25
3multi_head_attention_1/dropout/dropout/GreaterEqual?
+multi_head_attention_1/dropout/dropout/CastCast7multi_head_attention_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????--2-
+multi_head_attention_1/dropout/dropout/Cast?
,multi_head_attention_1/dropout/dropout/Mul_1Mul.multi_head_attention_1/dropout/dropout/Mul:z:0/multi_head_attention_1/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????--2.
,multi_head_attention_1/dropout/dropout/Mul_1?
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/dropout/Mul_1:z:0$multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:?????????- *
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/Einsum?
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp?
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????- *
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/Einsum?
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOp?
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2-
+multi_head_attention_1/attention_output/addw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMul/multi_head_attention_1/attention_output/add:z:0 dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:?????????- 2
dropout_2/dropout/Mul?
dropout_2/dropout/ShapeShape/multi_head_attention_1/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????- *
dtype0*

seed**
seed220
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????- 2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????- 2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????- 2
dropout_2/dropout/Mul_1n
addAddV2inputsdropout_2/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????- 2
add?
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indices?
"layer_normalization_2/moments/meanMeanadd:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2$
"layer_normalization_2/moments/mean?
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:?????????-2,
*layer_normalization_2/moments/StopGradient?
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????- 21
/layer_normalization_2/moments/SquaredDifference?
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indices?
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2(
&layer_normalization_2/moments/variance?
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_2/batchnorm/add/y?
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????-2%
#layer_normalization_2/batchnorm/add?
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????-2'
%layer_normalization_2/batchnorm/Rsqrt?
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOp?
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2%
#layer_normalization_2/batchnorm/mul?
%layer_normalization_2/batchnorm/mul_1Muladd:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_2/batchnorm/mul_1?
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_2/batchnorm/mul_2?
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_2/batchnorm/ReadVariableOp?
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????- 2%
#layer_normalization_2/batchnorm/sub?
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_2/batchnorm/add_1?
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_1/dense_2/Tensordot/ReadVariableOp?
#sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_2/Tensordot/axes?
#sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_2/Tensordot/free?
$sequential_1/dense_2/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/Shape?
,sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/GatherV2/axis?
'sequential_1/dense_2/Tensordot/GatherV2GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/free:output:05sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/GatherV2?
.sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_2/Tensordot/GatherV2_1/axis?
)sequential_1/dense_2/Tensordot/GatherV2_1GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/axes:output:07sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_2/Tensordot/GatherV2_1?
$sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_2/Tensordot/Const?
#sequential_1/dense_2/Tensordot/ProdProd0sequential_1/dense_2/Tensordot/GatherV2:output:0-sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_2/Tensordot/Prod?
&sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_1?
%sequential_1/dense_2/Tensordot/Prod_1Prod2sequential_1/dense_2/Tensordot/GatherV2_1:output:0/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_2/Tensordot/Prod_1?
*sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_2/Tensordot/concat/axis?
%sequential_1/dense_2/Tensordot/concatConcatV2,sequential_1/dense_2/Tensordot/free:output:0,sequential_1/dense_2/Tensordot/axes:output:03sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_2/Tensordot/concat?
$sequential_1/dense_2/Tensordot/stackPack,sequential_1/dense_2/Tensordot/Prod:output:0.sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/stack?
(sequential_1/dense_2/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0.sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2*
(sequential_1/dense_2/Tensordot/transpose?
&sequential_1/dense_2/Tensordot/ReshapeReshape,sequential_1/dense_2/Tensordot/transpose:y:0-sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&sequential_1/dense_2/Tensordot/Reshape?
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%sequential_1/dense_2/Tensordot/MatMul?
&sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_2?
,sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/concat_1/axis?
'sequential_1/dense_2/Tensordot/concat_1ConcatV20sequential_1/dense_2/Tensordot/GatherV2:output:0/sequential_1/dense_2/Tensordot/Const_2:output:05sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/concat_1?
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:00sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2 
sequential_1/dense_2/Tensordot?
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOp?
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2
sequential_1/dense_2/BiasAdd?
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????- 2
sequential_1/dense_2/Relu?
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_1/dense_3/Tensordot/ReadVariableOp?
#sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_3/Tensordot/axes?
#sequential_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_3/Tensordot/free?
$sequential_1/dense_3/Tensordot/ShapeShape'sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/Shape?
,sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/GatherV2/axis?
'sequential_1/dense_3/Tensordot/GatherV2GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/free:output:05sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/GatherV2?
.sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_3/Tensordot/GatherV2_1/axis?
)sequential_1/dense_3/Tensordot/GatherV2_1GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/axes:output:07sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_3/Tensordot/GatherV2_1?
$sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_3/Tensordot/Const?
#sequential_1/dense_3/Tensordot/ProdProd0sequential_1/dense_3/Tensordot/GatherV2:output:0-sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_3/Tensordot/Prod?
&sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_1?
%sequential_1/dense_3/Tensordot/Prod_1Prod2sequential_1/dense_3/Tensordot/GatherV2_1:output:0/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_3/Tensordot/Prod_1?
*sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_3/Tensordot/concat/axis?
%sequential_1/dense_3/Tensordot/concatConcatV2,sequential_1/dense_3/Tensordot/free:output:0,sequential_1/dense_3/Tensordot/axes:output:03sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_3/Tensordot/concat?
$sequential_1/dense_3/Tensordot/stackPack,sequential_1/dense_3/Tensordot/Prod:output:0.sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/stack?
(sequential_1/dense_3/Tensordot/transpose	Transpose'sequential_1/dense_2/Relu:activations:0.sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2*
(sequential_1/dense_3/Tensordot/transpose?
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_3/Tensordot/transpose:y:0-sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&sequential_1/dense_3/Tensordot/Reshape?
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%sequential_1/dense_3/Tensordot/MatMul?
&sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_2?
,sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/concat_1/axis?
'sequential_1/dense_3/Tensordot/concat_1ConcatV20sequential_1/dense_3/Tensordot/GatherV2:output:0/sequential_1/dense_3/Tensordot/Const_2:output:05sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/concat_1?
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:00sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2 
sequential_1/dense_3/Tensordot?
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOp?
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2
sequential_1/dense_3/BiasAddw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_3/dropout/Const?
dropout_3/dropout/MulMul%sequential_1/dense_3/BiasAdd:output:0 dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:?????????- 2
dropout_3/dropout/Mul?
dropout_3/dropout/ShapeShape%sequential_1/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????- *
dtype0*

seed**
seed220
.dropout_3/dropout/random_uniform/RandomUniform?
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_3/dropout/GreaterEqual/y?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????- 2 
dropout_3/dropout/GreaterEqual?
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????- 2
dropout_3/dropout/Cast?
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????- 2
dropout_3/dropout/Mul_1?
add_1AddV2)layer_normalization_2/batchnorm/add_1:z:0dropout_3/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????- 2
add_1?
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indices?
"layer_normalization_3/moments/meanMean	add_1:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2$
"layer_normalization_3/moments/mean?
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:?????????-2,
*layer_normalization_3/moments/StopGradient?
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????- 21
/layer_normalization_3/moments/SquaredDifference?
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indices?
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2(
&layer_normalization_3/moments/variance?
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_3/batchnorm/add/y?
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????-2%
#layer_normalization_3/batchnorm/add?
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????-2'
%layer_normalization_3/batchnorm/Rsqrt?
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOp?
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2%
#layer_normalization_3/batchnorm/mul?
%layer_normalization_3/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_3/batchnorm/mul_1?
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_3/batchnorm/mul_2?
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_3/batchnorm/ReadVariableOp?
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????- 2%
#layer_normalization_3/batchnorm/sub?
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_3/batchnorm/add_1?
IdentityIdentity)layer_normalization_3/batchnorm/add_1:z:0/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:?????????- ::::::::::::::::2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2?
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_1/key/add/ReadVariableOp-multi_head_attention_1/key/add/ReadVariableOp2r
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/query/add/ReadVariableOp/multi_head_attention_1/query/add/ReadVariableOp2v
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/value/add/ReadVariableOp/multi_head_attention_1/value/add/ReadVariableOp2v
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2^
-sequential_1/dense_2/Tensordot/ReadVariableOp-sequential_1/dense_2/Tensordot/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2^
-sequential_1/dense_3/Tensordot/ReadVariableOp-sequential_1/dense_3/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_88923

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????- :S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
?
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_88458

inputs
dense_2_88447
dense_2_88449
dense_3_88452
dense_3_88454
identity??dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_88447dense_2_88449*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_883372!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_88452dense_3_88454*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_883832!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????- ::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
??
?
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_88809

inputsF
Bmulti_head_attention_1_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_1_query_add_readvariableop_resourceD
@multi_head_attention_1_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_1_key_add_readvariableop_resourceF
Bmulti_head_attention_1_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_1_value_add_readvariableop_resourceQ
Mmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_1_attention_output_add_readvariableop_resource?
;layer_normalization_2_batchnorm_mul_readvariableop_resource;
7layer_normalization_2_batchnorm_readvariableop_resource:
6sequential_1_dense_2_tensordot_readvariableop_resource8
4sequential_1_dense_2_biasadd_readvariableop_resource:
6sequential_1_dense_3_tensordot_readvariableop_resource8
4sequential_1_dense_3_biasadd_readvariableop_resource?
;layer_normalization_3_batchnorm_mul_readvariableop_resource;
7layer_normalization_3_batchnorm_readvariableop_resource
identity??.layer_normalization_2/batchnorm/ReadVariableOp?2layer_normalization_2/batchnorm/mul/ReadVariableOp?.layer_normalization_3/batchnorm/ReadVariableOp?2layer_normalization_3/batchnorm/mul/ReadVariableOp?:multi_head_attention_1/attention_output/add/ReadVariableOp?Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp?-multi_head_attention_1/key/add/ReadVariableOp?7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp?/multi_head_attention_1/query/add/ReadVariableOp?9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?/multi_head_attention_1/value/add/ReadVariableOp?9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp?+sequential_1/dense_2/BiasAdd/ReadVariableOp?-sequential_1/dense_2/Tensordot/ReadVariableOp?+sequential_1/dense_3/BiasAdd/ReadVariableOp?-sequential_1/dense_3/Tensordot/ReadVariableOp?
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?
*multi_head_attention_1/query/einsum/EinsumEinsuminputsAmulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/Einsum?
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/query/add/ReadVariableOp?
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 2"
 multi_head_attention_1/query/add?
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp?
(multi_head_attention_1/key/einsum/EinsumEinsuminputs?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/Einsum?
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_1/key/add/ReadVariableOp?
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 2 
multi_head_attention_1/key/add?
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp?
*multi_head_attention_1/value/einsum/EinsumEinsuminputsAmulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/Einsum?
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/value/add/ReadVariableOp?
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 2"
 multi_head_attention_1/value/add?
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>2
multi_head_attention_1/Mul/y?
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:?????????- 2
multi_head_attention_1/Mul?
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:?????????--*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/Einsum?
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????--2(
&multi_head_attention_1/softmax/Softmax?
'multi_head_attention_1/dropout/IdentityIdentity0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????--2)
'multi_head_attention_1/dropout/Identity?
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/Identity:output:0$multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:?????????- *
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/Einsum?
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp?
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????- *
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/Einsum?
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOp?
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2-
+multi_head_attention_1/attention_output/add?
dropout_2/IdentityIdentity/multi_head_attention_1/attention_output/add:z:0*
T0*+
_output_shapes
:?????????- 2
dropout_2/Identityn
addAddV2inputsdropout_2/Identity:output:0*
T0*+
_output_shapes
:?????????- 2
add?
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indices?
"layer_normalization_2/moments/meanMeanadd:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2$
"layer_normalization_2/moments/mean?
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:?????????-2,
*layer_normalization_2/moments/StopGradient?
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????- 21
/layer_normalization_2/moments/SquaredDifference?
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indices?
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2(
&layer_normalization_2/moments/variance?
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_2/batchnorm/add/y?
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????-2%
#layer_normalization_2/batchnorm/add?
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????-2'
%layer_normalization_2/batchnorm/Rsqrt?
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOp?
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2%
#layer_normalization_2/batchnorm/mul?
%layer_normalization_2/batchnorm/mul_1Muladd:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_2/batchnorm/mul_1?
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_2/batchnorm/mul_2?
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_2/batchnorm/ReadVariableOp?
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????- 2%
#layer_normalization_2/batchnorm/sub?
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_2/batchnorm/add_1?
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_1/dense_2/Tensordot/ReadVariableOp?
#sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_2/Tensordot/axes?
#sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_2/Tensordot/free?
$sequential_1/dense_2/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/Shape?
,sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/GatherV2/axis?
'sequential_1/dense_2/Tensordot/GatherV2GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/free:output:05sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/GatherV2?
.sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_2/Tensordot/GatherV2_1/axis?
)sequential_1/dense_2/Tensordot/GatherV2_1GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/axes:output:07sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_2/Tensordot/GatherV2_1?
$sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_2/Tensordot/Const?
#sequential_1/dense_2/Tensordot/ProdProd0sequential_1/dense_2/Tensordot/GatherV2:output:0-sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_2/Tensordot/Prod?
&sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_1?
%sequential_1/dense_2/Tensordot/Prod_1Prod2sequential_1/dense_2/Tensordot/GatherV2_1:output:0/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_2/Tensordot/Prod_1?
*sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_2/Tensordot/concat/axis?
%sequential_1/dense_2/Tensordot/concatConcatV2,sequential_1/dense_2/Tensordot/free:output:0,sequential_1/dense_2/Tensordot/axes:output:03sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_2/Tensordot/concat?
$sequential_1/dense_2/Tensordot/stackPack,sequential_1/dense_2/Tensordot/Prod:output:0.sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/stack?
(sequential_1/dense_2/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0.sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2*
(sequential_1/dense_2/Tensordot/transpose?
&sequential_1/dense_2/Tensordot/ReshapeReshape,sequential_1/dense_2/Tensordot/transpose:y:0-sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&sequential_1/dense_2/Tensordot/Reshape?
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%sequential_1/dense_2/Tensordot/MatMul?
&sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_2?
,sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/concat_1/axis?
'sequential_1/dense_2/Tensordot/concat_1ConcatV20sequential_1/dense_2/Tensordot/GatherV2:output:0/sequential_1/dense_2/Tensordot/Const_2:output:05sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/concat_1?
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:00sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2 
sequential_1/dense_2/Tensordot?
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOp?
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2
sequential_1/dense_2/BiasAdd?
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????- 2
sequential_1/dense_2/Relu?
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_1/dense_3/Tensordot/ReadVariableOp?
#sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_3/Tensordot/axes?
#sequential_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_3/Tensordot/free?
$sequential_1/dense_3/Tensordot/ShapeShape'sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/Shape?
,sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/GatherV2/axis?
'sequential_1/dense_3/Tensordot/GatherV2GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/free:output:05sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/GatherV2?
.sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_3/Tensordot/GatherV2_1/axis?
)sequential_1/dense_3/Tensordot/GatherV2_1GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/axes:output:07sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_3/Tensordot/GatherV2_1?
$sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_3/Tensordot/Const?
#sequential_1/dense_3/Tensordot/ProdProd0sequential_1/dense_3/Tensordot/GatherV2:output:0-sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_3/Tensordot/Prod?
&sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_1?
%sequential_1/dense_3/Tensordot/Prod_1Prod2sequential_1/dense_3/Tensordot/GatherV2_1:output:0/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_3/Tensordot/Prod_1?
*sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_3/Tensordot/concat/axis?
%sequential_1/dense_3/Tensordot/concatConcatV2,sequential_1/dense_3/Tensordot/free:output:0,sequential_1/dense_3/Tensordot/axes:output:03sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_3/Tensordot/concat?
$sequential_1/dense_3/Tensordot/stackPack,sequential_1/dense_3/Tensordot/Prod:output:0.sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/stack?
(sequential_1/dense_3/Tensordot/transpose	Transpose'sequential_1/dense_2/Relu:activations:0.sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2*
(sequential_1/dense_3/Tensordot/transpose?
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_3/Tensordot/transpose:y:0-sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&sequential_1/dense_3/Tensordot/Reshape?
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%sequential_1/dense_3/Tensordot/MatMul?
&sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_2?
,sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/concat_1/axis?
'sequential_1/dense_3/Tensordot/concat_1ConcatV20sequential_1/dense_3/Tensordot/GatherV2:output:0/sequential_1/dense_3/Tensordot/Const_2:output:05sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/concat_1?
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:00sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2 
sequential_1/dense_3/Tensordot?
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOp?
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2
sequential_1/dense_3/BiasAdd?
dropout_3/IdentityIdentity%sequential_1/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????- 2
dropout_3/Identity?
add_1AddV2)layer_normalization_2/batchnorm/add_1:z:0dropout_3/Identity:output:0*
T0*+
_output_shapes
:?????????- 2
add_1?
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indices?
"layer_normalization_3/moments/meanMean	add_1:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2$
"layer_normalization_3/moments/mean?
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:?????????-2,
*layer_normalization_3/moments/StopGradient?
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????- 21
/layer_normalization_3/moments/SquaredDifference?
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indices?
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2(
&layer_normalization_3/moments/variance?
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_3/batchnorm/add/y?
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????-2%
#layer_normalization_3/batchnorm/add?
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????-2'
%layer_normalization_3/batchnorm/Rsqrt?
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOp?
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2%
#layer_normalization_3/batchnorm/mul?
%layer_normalization_3/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_3/batchnorm/mul_1?
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_3/batchnorm/mul_2?
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_3/batchnorm/ReadVariableOp?
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????- 2%
#layer_normalization_3/batchnorm/sub?
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????- 2'
%layer_normalization_3/batchnorm/add_1?
IdentityIdentity)layer_normalization_3/batchnorm/add_1:z:0/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:?????????- ::::::::::::::::2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2?
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_1/key/add/ReadVariableOp-multi_head_attention_1/key/add/ReadVariableOp2r
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/query/add/ReadVariableOp/multi_head_attention_1/query/add/ReadVariableOp2v
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/value/add/ReadVariableOp/multi_head_attention_1/value/add/ReadVariableOp2v
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2^
-sequential_1/dense_2/Tensordot/ReadVariableOp-sequential_1/dense_2/Tensordot/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2^
-sequential_1/dense_3/Tensordot/ReadVariableOp-sequential_1/dense_3/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
?
h
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_88296

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+???????????????????????????*
ksize	
?*
paddingVALID*
strides	
?2	
AvgPool?
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_5_layer_call_fn_90452

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_890422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
E
)__inference_dropout_5_layer_call_fn_90457

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_890472
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?}
?
__inference__traced_save_90881
file_prefix-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	P
Lsavev2_token_and_position_embedding_embedding_embeddings_read_readvariableopR
Nsavev2_token_and_position_embedding_embedding_1_embeddings_read_readvariableopV
Rsavev2_transformer_block_1_multi_head_attention_1_query_kernel_read_readvariableopT
Psavev2_transformer_block_1_multi_head_attention_1_query_bias_read_readvariableopT
Psavev2_transformer_block_1_multi_head_attention_1_key_kernel_read_readvariableopR
Nsavev2_transformer_block_1_multi_head_attention_1_key_bias_read_readvariableopV
Rsavev2_transformer_block_1_multi_head_attention_1_value_kernel_read_readvariableopT
Psavev2_transformer_block_1_multi_head_attention_1_value_bias_read_readvariableopa
]savev2_transformer_block_1_multi_head_attention_1_attention_output_kernel_read_readvariableop_
[savev2_transformer_block_1_multi_head_attention_1_attention_output_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableopN
Jsavev2_transformer_block_1_layer_normalization_2_gamma_read_readvariableopM
Isavev2_transformer_block_1_layer_normalization_2_beta_read_readvariableopN
Jsavev2_transformer_block_1_layer_normalization_3_gamma_read_readvariableopM
Isavev2_transformer_block_1_layer_normalization_3_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop:
6savev2_sgd_dense_4_kernel_momentum_read_readvariableop8
4savev2_sgd_dense_4_bias_momentum_read_readvariableop:
6savev2_sgd_dense_5_kernel_momentum_read_readvariableop8
4savev2_sgd_dense_5_bias_momentum_read_readvariableop:
6savev2_sgd_dense_6_kernel_momentum_read_readvariableop8
4savev2_sgd_dense_6_bias_momentum_read_readvariableop]
Ysavev2_sgd_token_and_position_embedding_embedding_embeddings_momentum_read_readvariableop_
[savev2_sgd_token_and_position_embedding_embedding_1_embeddings_momentum_read_readvariableopc
_savev2_sgd_transformer_block_1_multi_head_attention_1_query_kernel_momentum_read_readvariableopa
]savev2_sgd_transformer_block_1_multi_head_attention_1_query_bias_momentum_read_readvariableopa
]savev2_sgd_transformer_block_1_multi_head_attention_1_key_kernel_momentum_read_readvariableop_
[savev2_sgd_transformer_block_1_multi_head_attention_1_key_bias_momentum_read_readvariableopc
_savev2_sgd_transformer_block_1_multi_head_attention_1_value_kernel_momentum_read_readvariableopa
]savev2_sgd_transformer_block_1_multi_head_attention_1_value_bias_momentum_read_readvariableopn
jsavev2_sgd_transformer_block_1_multi_head_attention_1_attention_output_kernel_momentum_read_readvariableopl
hsavev2_sgd_transformer_block_1_multi_head_attention_1_attention_output_bias_momentum_read_readvariableop:
6savev2_sgd_dense_2_kernel_momentum_read_readvariableop8
4savev2_sgd_dense_2_bias_momentum_read_readvariableop:
6savev2_sgd_dense_3_kernel_momentum_read_readvariableop8
4savev2_sgd_dense_3_bias_momentum_read_readvariableop[
Wsavev2_sgd_transformer_block_1_layer_normalization_2_gamma_momentum_read_readvariableopZ
Vsavev2_sgd_transformer_block_1_layer_normalization_2_beta_momentum_read_readvariableop[
Wsavev2_sgd_transformer_block_1_layer_normalization_3_gamma_momentum_read_readvariableopZ
Vsavev2_sgd_transformer_block_1_layer_normalization_3_beta_momentum_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
value?B?7B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/8/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableopLsavev2_token_and_position_embedding_embedding_embeddings_read_readvariableopNsavev2_token_and_position_embedding_embedding_1_embeddings_read_readvariableopRsavev2_transformer_block_1_multi_head_attention_1_query_kernel_read_readvariableopPsavev2_transformer_block_1_multi_head_attention_1_query_bias_read_readvariableopPsavev2_transformer_block_1_multi_head_attention_1_key_kernel_read_readvariableopNsavev2_transformer_block_1_multi_head_attention_1_key_bias_read_readvariableopRsavev2_transformer_block_1_multi_head_attention_1_value_kernel_read_readvariableopPsavev2_transformer_block_1_multi_head_attention_1_value_bias_read_readvariableop]savev2_transformer_block_1_multi_head_attention_1_attention_output_kernel_read_readvariableop[savev2_transformer_block_1_multi_head_attention_1_attention_output_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableopJsavev2_transformer_block_1_layer_normalization_2_gamma_read_readvariableopIsavev2_transformer_block_1_layer_normalization_2_beta_read_readvariableopJsavev2_transformer_block_1_layer_normalization_3_gamma_read_readvariableopIsavev2_transformer_block_1_layer_normalization_3_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_sgd_dense_4_kernel_momentum_read_readvariableop4savev2_sgd_dense_4_bias_momentum_read_readvariableop6savev2_sgd_dense_5_kernel_momentum_read_readvariableop4savev2_sgd_dense_5_bias_momentum_read_readvariableop6savev2_sgd_dense_6_kernel_momentum_read_readvariableop4savev2_sgd_dense_6_bias_momentum_read_readvariableopYsavev2_sgd_token_and_position_embedding_embedding_embeddings_momentum_read_readvariableop[savev2_sgd_token_and_position_embedding_embedding_1_embeddings_momentum_read_readvariableop_savev2_sgd_transformer_block_1_multi_head_attention_1_query_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_1_multi_head_attention_1_query_bias_momentum_read_readvariableop]savev2_sgd_transformer_block_1_multi_head_attention_1_key_kernel_momentum_read_readvariableop[savev2_sgd_transformer_block_1_multi_head_attention_1_key_bias_momentum_read_readvariableop_savev2_sgd_transformer_block_1_multi_head_attention_1_value_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_1_multi_head_attention_1_value_bias_momentum_read_readvariableopjsavev2_sgd_transformer_block_1_multi_head_attention_1_attention_output_kernel_momentum_read_readvariableophsavev2_sgd_transformer_block_1_multi_head_attention_1_attention_output_bias_momentum_read_readvariableop6savev2_sgd_dense_2_kernel_momentum_read_readvariableop4savev2_sgd_dense_2_bias_momentum_read_readvariableop6savev2_sgd_dense_3_kernel_momentum_read_readvariableop4savev2_sgd_dense_3_bias_momentum_read_readvariableopWsavev2_sgd_transformer_block_1_layer_normalization_2_gamma_momentum_read_readvariableopVsavev2_sgd_transformer_block_1_layer_normalization_2_beta_momentum_read_readvariableopWsavev2_sgd_transformer_block_1_layer_normalization_3_gamma_momentum_read_readvariableopVsavev2_sgd_transformer_block_1_layer_normalization_3_beta_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *E
dtypes;
927	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :( : :  : : :: : : : : :	?i :  : :  : :  : :  : :  : :  : : : : : : : :( : :  : : :: :	?i :  : :  : :  : :  : :  : :  : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:( : 

_output_shapes
: :$ 

_output_shapes

:  : 
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
: :$ 

_output_shapes

: :%!

_output_shapes
:	?i :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:( :  

_output_shapes
: :$! 

_output_shapes

:  : "

_output_shapes
: :$# 

_output_shapes

: : $

_output_shapes
::$% 

_output_shapes

: :%&!

_output_shapes
:	?i :('$
"
_output_shapes
:  :$( 

_output_shapes

: :()$
"
_output_shapes
:  :$* 

_output_shapes

: :(+$
"
_output_shapes
:  :$, 

_output_shapes

: :(-$
"
_output_shapes
:  : .

_output_shapes
: :$/ 

_output_shapes

:  : 0

_output_shapes
: :$1 

_output_shapes

:  : 2

_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: : 5

_output_shapes
: : 6

_output_shapes
: :7

_output_shapes
: 
?
?
,__inference_sequential_1_layer_call_fn_90616

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_884582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????- ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_88287
input_1
input_2I
Emodel_token_and_position_embedding_embedding_1_embedding_lookup_88121G
Cmodel_token_and_position_embedding_embedding_embedding_lookup_88127`
\model_transformer_block_1_multi_head_attention_1_query_einsum_einsum_readvariableop_resourceV
Rmodel_transformer_block_1_multi_head_attention_1_query_add_readvariableop_resource^
Zmodel_transformer_block_1_multi_head_attention_1_key_einsum_einsum_readvariableop_resourceT
Pmodel_transformer_block_1_multi_head_attention_1_key_add_readvariableop_resource`
\model_transformer_block_1_multi_head_attention_1_value_einsum_einsum_readvariableop_resourceV
Rmodel_transformer_block_1_multi_head_attention_1_value_add_readvariableop_resourcek
gmodel_transformer_block_1_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resourcea
]model_transformer_block_1_multi_head_attention_1_attention_output_add_readvariableop_resourceY
Umodel_transformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resourceU
Qmodel_transformer_block_1_layer_normalization_2_batchnorm_readvariableop_resourceT
Pmodel_transformer_block_1_sequential_1_dense_2_tensordot_readvariableop_resourceR
Nmodel_transformer_block_1_sequential_1_dense_2_biasadd_readvariableop_resourceT
Pmodel_transformer_block_1_sequential_1_dense_3_tensordot_readvariableop_resourceR
Nmodel_transformer_block_1_sequential_1_dense_3_biasadd_readvariableop_resourceY
Umodel_transformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resourceU
Qmodel_transformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource0
,model_dense_4_matmul_readvariableop_resource1
-model_dense_4_biasadd_readvariableop_resource0
,model_dense_5_matmul_readvariableop_resource1
-model_dense_5_biasadd_readvariableop_resource0
,model_dense_6_matmul_readvariableop_resource1
-model_dense_6_biasadd_readvariableop_resource
identity??$model/dense_4/BiasAdd/ReadVariableOp?#model/dense_4/MatMul/ReadVariableOp?$model/dense_5/BiasAdd/ReadVariableOp?#model/dense_5/MatMul/ReadVariableOp?$model/dense_6/BiasAdd/ReadVariableOp?#model/dense_6/MatMul/ReadVariableOp?=model/token_and_position_embedding/embedding/embedding_lookup??model/token_and_position_embedding/embedding_1/embedding_lookup?Hmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp?Lmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp?Hmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp?Lmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp?Tmodel/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp?^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp?Gmodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOp?Qmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp?Imodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOp?Smodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?Imodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOp?Smodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp?Emodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp?Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp?Emodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp?Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp?
(model/token_and_position_embedding/ShapeShapeinput_1*
T0*
_output_shapes
:2*
(model/token_and_position_embedding/Shape?
6model/token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????28
6model/token_and_position_embedding/strided_slice/stack?
8model/token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8model/token_and_position_embedding/strided_slice/stack_1?
8model/token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/token_and_position_embedding/strided_slice/stack_2?
0model/token_and_position_embedding/strided_sliceStridedSlice1model/token_and_position_embedding/Shape:output:0?model/token_and_position_embedding/strided_slice/stack:output:0Amodel/token_and_position_embedding/strided_slice/stack_1:output:0Amodel/token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0model/token_and_position_embedding/strided_slice?
.model/token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : 20
.model/token_and_position_embedding/range/start?
.model/token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :20
.model/token_and_position_embedding/range/delta?
(model/token_and_position_embedding/rangeRange7model/token_and_position_embedding/range/start:output:09model/token_and_position_embedding/strided_slice:output:07model/token_and_position_embedding/range/delta:output:0*#
_output_shapes
:?????????2*
(model/token_and_position_embedding/range?
?model/token_and_position_embedding/embedding_1/embedding_lookupResourceGatherEmodel_token_and_position_embedding_embedding_1_embedding_lookup_881211model/token_and_position_embedding/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*X
_classN
LJloc:@model/token_and_position_embedding/embedding_1/embedding_lookup/88121*'
_output_shapes
:????????? *
dtype02A
?model/token_and_position_embedding/embedding_1/embedding_lookup?
Hmodel/token_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityHmodel/token_and_position_embedding/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@model/token_and_position_embedding/embedding_1/embedding_lookup/88121*'
_output_shapes
:????????? 2J
Hmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity?
Jmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityQmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2L
Jmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity_1?
1model/token_and_position_embedding/embedding/CastCastinput_1*

DstT0*

SrcT0*(
_output_shapes
:??????????i23
1model/token_and_position_embedding/embedding/Cast?
=model/token_and_position_embedding/embedding/embedding_lookupResourceGatherCmodel_token_and_position_embedding_embedding_embedding_lookup_881275model/token_and_position_embedding/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*V
_classL
JHloc:@model/token_and_position_embedding/embedding/embedding_lookup/88127*,
_output_shapes
:??????????i *
dtype02?
=model/token_and_position_embedding/embedding/embedding_lookup?
Fmodel/token_and_position_embedding/embedding/embedding_lookup/IdentityIdentityFmodel/token_and_position_embedding/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@model/token_and_position_embedding/embedding/embedding_lookup/88127*,
_output_shapes
:??????????i 2H
Fmodel/token_and_position_embedding/embedding/embedding_lookup/Identity?
Hmodel/token_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityOmodel/token_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????i 2J
Hmodel/token_and_position_embedding/embedding/embedding_lookup/Identity_1?
&model/token_and_position_embedding/addAddV2Qmodel/token_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Smodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????i 2(
&model/token_and_position_embedding/add?
&model/average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&model/average_pooling1d/ExpandDims/dim?
"model/average_pooling1d/ExpandDims
ExpandDims*model/token_and_position_embedding/add:z:0/model/average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????i 2$
"model/average_pooling1d/ExpandDims?
model/average_pooling1d/AvgPoolAvgPool+model/average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:?????????- *
ksize	
?*
paddingVALID*
strides	
?2!
model/average_pooling1d/AvgPool?
model/average_pooling1d/SqueezeSqueeze(model/average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:?????????- *
squeeze_dims
2!
model/average_pooling1d/Squeeze?
Smodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOp\model_transformer_block_1_multi_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02U
Smodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?
Dmodel/transformer_block_1/multi_head_attention_1/query/einsum/EinsumEinsum(model/average_pooling1d/Squeeze:output:0[model/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2F
Dmodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum?
Imodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpReadVariableOpRmodel_transformer_block_1_multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype02K
Imodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOp?
:model/transformer_block_1/multi_head_attention_1/query/addAddV2Mmodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum:output:0Qmodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 2<
:model/transformer_block_1/multi_head_attention_1/query/add?
Qmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOpZmodel_transformer_block_1_multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02S
Qmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp?
Bmodel/transformer_block_1/multi_head_attention_1/key/einsum/EinsumEinsum(model/average_pooling1d/Squeeze:output:0Ymodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2D
Bmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum?
Gmodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOpReadVariableOpPmodel_transformer_block_1_multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02I
Gmodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOp?
8model/transformer_block_1/multi_head_attention_1/key/addAddV2Kmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum:output:0Omodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 2:
8model/transformer_block_1/multi_head_attention_1/key/add?
Smodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOp\model_transformer_block_1_multi_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02U
Smodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp?
Dmodel/transformer_block_1/multi_head_attention_1/value/einsum/EinsumEinsum(model/average_pooling1d/Squeeze:output:0[model/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????- *
equationabc,cde->abde2F
Dmodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum?
Imodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpReadVariableOpRmodel_transformer_block_1_multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype02K
Imodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOp?
:model/transformer_block_1/multi_head_attention_1/value/addAddV2Mmodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum:output:0Qmodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????- 2<
:model/transformer_block_1/multi_head_attention_1/value/add?
6model/transformer_block_1/multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?5>28
6model/transformer_block_1/multi_head_attention_1/Mul/y?
4model/transformer_block_1/multi_head_attention_1/MulMul>model/transformer_block_1/multi_head_attention_1/query/add:z:0?model/transformer_block_1/multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:?????????- 26
4model/transformer_block_1/multi_head_attention_1/Mul?
>model/transformer_block_1/multi_head_attention_1/einsum/EinsumEinsum<model/transformer_block_1/multi_head_attention_1/key/add:z:08model/transformer_block_1/multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:?????????--*
equationaecd,abcd->acbe2@
>model/transformer_block_1/multi_head_attention_1/einsum/Einsum?
@model/transformer_block_1/multi_head_attention_1/softmax/SoftmaxSoftmaxGmodel/transformer_block_1/multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????--2B
@model/transformer_block_1/multi_head_attention_1/softmax/Softmax?
Amodel/transformer_block_1/multi_head_attention_1/dropout/IdentityIdentityJmodel/transformer_block_1/multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????--2C
Amodel/transformer_block_1/multi_head_attention_1/dropout/Identity?
@model/transformer_block_1/multi_head_attention_1/einsum_1/EinsumEinsumJmodel/transformer_block_1/multi_head_attention_1/dropout/Identity:output:0>model/transformer_block_1/multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:?????????- *
equationacbe,aecd->abcd2B
@model/transformer_block_1/multi_head_attention_1/einsum_1/Einsum?
^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_transformer_block_1_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02`
^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp?
Omodel/transformer_block_1/multi_head_attention_1/attention_output/einsum/EinsumEinsumImodel/transformer_block_1/multi_head_attention_1/einsum_1/Einsum:output:0fmodel/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????- *
equationabcd,cde->abe2Q
Omodel/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum?
Tmodel/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOp]model_transformer_block_1_multi_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02V
Tmodel/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp?
Emodel/transformer_block_1/multi_head_attention_1/attention_output/addAddV2Xmodel/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum:output:0\model/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2G
Emodel/transformer_block_1/multi_head_attention_1/attention_output/add?
,model/transformer_block_1/dropout_2/IdentityIdentityImodel/transformer_block_1/multi_head_attention_1/attention_output/add:z:0*
T0*+
_output_shapes
:?????????- 2.
,model/transformer_block_1/dropout_2/Identity?
model/transformer_block_1/addAddV2(model/average_pooling1d/Squeeze:output:05model/transformer_block_1/dropout_2/Identity:output:0*
T0*+
_output_shapes
:?????????- 2
model/transformer_block_1/add?
Nmodel/transformer_block_1/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
Nmodel/transformer_block_1/layer_normalization_2/moments/mean/reduction_indices?
<model/transformer_block_1/layer_normalization_2/moments/meanMean!model/transformer_block_1/add:z:0Wmodel/transformer_block_1/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2>
<model/transformer_block_1/layer_normalization_2/moments/mean?
Dmodel/transformer_block_1/layer_normalization_2/moments/StopGradientStopGradientEmodel/transformer_block_1/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:?????????-2F
Dmodel/transformer_block_1/layer_normalization_2/moments/StopGradient?
Imodel/transformer_block_1/layer_normalization_2/moments/SquaredDifferenceSquaredDifference!model/transformer_block_1/add:z:0Mmodel/transformer_block_1/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????- 2K
Imodel/transformer_block_1/layer_normalization_2/moments/SquaredDifference?
Rmodel/transformer_block_1/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2T
Rmodel/transformer_block_1/layer_normalization_2/moments/variance/reduction_indices?
@model/transformer_block_1/layer_normalization_2/moments/varianceMeanMmodel/transformer_block_1/layer_normalization_2/moments/SquaredDifference:z:0[model/transformer_block_1/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2B
@model/transformer_block_1/layer_normalization_2/moments/variance?
?model/transformer_block_1/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52A
?model/transformer_block_1/layer_normalization_2/batchnorm/add/y?
=model/transformer_block_1/layer_normalization_2/batchnorm/addAddV2Imodel/transformer_block_1/layer_normalization_2/moments/variance:output:0Hmodel/transformer_block_1/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????-2?
=model/transformer_block_1/layer_normalization_2/batchnorm/add?
?model/transformer_block_1/layer_normalization_2/batchnorm/RsqrtRsqrtAmodel/transformer_block_1/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????-2A
?model/transformer_block_1/layer_normalization_2/batchnorm/Rsqrt?
Lmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpUmodel_transformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02N
Lmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp?
=model/transformer_block_1/layer_normalization_2/batchnorm/mulMulCmodel/transformer_block_1/layer_normalization_2/batchnorm/Rsqrt:y:0Tmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2?
=model/transformer_block_1/layer_normalization_2/batchnorm/mul?
?model/transformer_block_1/layer_normalization_2/batchnorm/mul_1Mul!model/transformer_block_1/add:z:0Amodel/transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2A
?model/transformer_block_1/layer_normalization_2/batchnorm/mul_1?
?model/transformer_block_1/layer_normalization_2/batchnorm/mul_2MulEmodel/transformer_block_1/layer_normalization_2/moments/mean:output:0Amodel/transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2A
?model/transformer_block_1/layer_normalization_2/batchnorm/mul_2?
Hmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpQmodel_transformer_block_1_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp?
=model/transformer_block_1/layer_normalization_2/batchnorm/subSubPmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp:value:0Cmodel/transformer_block_1/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????- 2?
=model/transformer_block_1/layer_normalization_2/batchnorm/sub?
?model/transformer_block_1/layer_normalization_2/batchnorm/add_1AddV2Cmodel/transformer_block_1/layer_normalization_2/batchnorm/mul_1:z:0Amodel/transformer_block_1/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????- 2A
?model/transformer_block_1/layer_normalization_2/batchnorm/add_1?
Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOpPmodel_transformer_block_1_sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02I
Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp?
=model/transformer_block_1/sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2?
=model/transformer_block_1/sequential_1/dense_2/Tensordot/axes?
=model/transformer_block_1/sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model/transformer_block_1/sequential_1/dense_2/Tensordot/free?
>model/transformer_block_1/sequential_1/dense_2/Tensordot/ShapeShapeCmodel/transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2@
>model/transformer_block_1/sequential_1/dense_2/Tensordot/Shape?
Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axis?
Amodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2GatherV2Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0Omodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Amodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2?
Hmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axis?
Cmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1GatherV2Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Qmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1?
>model/transformer_block_1/sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2@
>model/transformer_block_1/sequential_1/dense_2/Tensordot/Const?
=model/transformer_block_1/sequential_1/dense_2/Tensordot/ProdProdJmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2?
=model/transformer_block_1/sequential_1/dense_2/Tensordot/Prod?
@model/transformer_block_1/sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block_1/sequential_1/dense_2/Tensordot/Const_1?
?model/transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1ProdLmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1:output:0Imodel/transformer_block_1/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2A
?model/transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1?
Dmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat/axis?
?model/transformer_block_1/sequential_1/dense_2/Tensordot/concatConcatV2Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Mmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2A
?model/transformer_block_1/sequential_1/dense_2/Tensordot/concat?
>model/transformer_block_1/sequential_1/dense_2/Tensordot/stackPackFmodel/transformer_block_1/sequential_1/dense_2/Tensordot/Prod:output:0Hmodel/transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2@
>model/transformer_block_1/sequential_1/dense_2/Tensordot/stack?
Bmodel/transformer_block_1/sequential_1/dense_2/Tensordot/transpose	TransposeCmodel/transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0Hmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2D
Bmodel/transformer_block_1/sequential_1/dense_2/Tensordot/transpose?
@model/transformer_block_1/sequential_1/dense_2/Tensordot/ReshapeReshapeFmodel/transformer_block_1/sequential_1/dense_2/Tensordot/transpose:y:0Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2B
@model/transformer_block_1/sequential_1/dense_2/Tensordot/Reshape?
?model/transformer_block_1/sequential_1/dense_2/Tensordot/MatMulMatMulImodel/transformer_block_1/sequential_1/dense_2/Tensordot/Reshape:output:0Omodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2A
?model/transformer_block_1/sequential_1/dense_2/Tensordot/MatMul?
@model/transformer_block_1/sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block_1/sequential_1/dense_2/Tensordot/Const_2?
Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axis?
Amodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1ConcatV2Jmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Imodel/transformer_block_1/sequential_1/dense_2/Tensordot/Const_2:output:0Omodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1?
8model/transformer_block_1/sequential_1/dense_2/TensordotReshapeImodel/transformer_block_1/sequential_1/dense_2/Tensordot/MatMul:product:0Jmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2:
8model/transformer_block_1/sequential_1/dense_2/Tensordot?
Emodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOpNmodel_transformer_block_1_sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02G
Emodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp?
6model/transformer_block_1/sequential_1/dense_2/BiasAddBiasAddAmodel/transformer_block_1/sequential_1/dense_2/Tensordot:output:0Mmodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 28
6model/transformer_block_1/sequential_1/dense_2/BiasAdd?
3model/transformer_block_1/sequential_1/dense_2/ReluRelu?model/transformer_block_1/sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????- 25
3model/transformer_block_1/sequential_1/dense_2/Relu?
Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOpPmodel_transformer_block_1_sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02I
Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp?
=model/transformer_block_1/sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2?
=model/transformer_block_1/sequential_1/dense_3/Tensordot/axes?
=model/transformer_block_1/sequential_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model/transformer_block_1/sequential_1/dense_3/Tensordot/free?
>model/transformer_block_1/sequential_1/dense_3/Tensordot/ShapeShapeAmodel/transformer_block_1/sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2@
>model/transformer_block_1/sequential_1/dense_3/Tensordot/Shape?
Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axis?
Amodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2GatherV2Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0Omodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Amodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2?
Hmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axis?
Cmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1GatherV2Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Qmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1?
>model/transformer_block_1/sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2@
>model/transformer_block_1/sequential_1/dense_3/Tensordot/Const?
=model/transformer_block_1/sequential_1/dense_3/Tensordot/ProdProdJmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2?
=model/transformer_block_1/sequential_1/dense_3/Tensordot/Prod?
@model/transformer_block_1/sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block_1/sequential_1/dense_3/Tensordot/Const_1?
?model/transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1ProdLmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1:output:0Imodel/transformer_block_1/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2A
?model/transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1?
Dmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat/axis?
?model/transformer_block_1/sequential_1/dense_3/Tensordot/concatConcatV2Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Mmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2A
?model/transformer_block_1/sequential_1/dense_3/Tensordot/concat?
>model/transformer_block_1/sequential_1/dense_3/Tensordot/stackPackFmodel/transformer_block_1/sequential_1/dense_3/Tensordot/Prod:output:0Hmodel/transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2@
>model/transformer_block_1/sequential_1/dense_3/Tensordot/stack?
Bmodel/transformer_block_1/sequential_1/dense_3/Tensordot/transpose	TransposeAmodel/transformer_block_1/sequential_1/dense_2/Relu:activations:0Hmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2D
Bmodel/transformer_block_1/sequential_1/dense_3/Tensordot/transpose?
@model/transformer_block_1/sequential_1/dense_3/Tensordot/ReshapeReshapeFmodel/transformer_block_1/sequential_1/dense_3/Tensordot/transpose:y:0Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2B
@model/transformer_block_1/sequential_1/dense_3/Tensordot/Reshape?
?model/transformer_block_1/sequential_1/dense_3/Tensordot/MatMulMatMulImodel/transformer_block_1/sequential_1/dense_3/Tensordot/Reshape:output:0Omodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2A
?model/transformer_block_1/sequential_1/dense_3/Tensordot/MatMul?
@model/transformer_block_1/sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block_1/sequential_1/dense_3/Tensordot/Const_2?
Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axis?
Amodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1ConcatV2Jmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Imodel/transformer_block_1/sequential_1/dense_3/Tensordot/Const_2:output:0Omodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1?
8model/transformer_block_1/sequential_1/dense_3/TensordotReshapeImodel/transformer_block_1/sequential_1/dense_3/Tensordot/MatMul:product:0Jmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2:
8model/transformer_block_1/sequential_1/dense_3/Tensordot?
Emodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOpNmodel_transformer_block_1_sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02G
Emodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp?
6model/transformer_block_1/sequential_1/dense_3/BiasAddBiasAddAmodel/transformer_block_1/sequential_1/dense_3/Tensordot:output:0Mmodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 28
6model/transformer_block_1/sequential_1/dense_3/BiasAdd?
,model/transformer_block_1/dropout_3/IdentityIdentity?model/transformer_block_1/sequential_1/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????- 2.
,model/transformer_block_1/dropout_3/Identity?
model/transformer_block_1/add_1AddV2Cmodel/transformer_block_1/layer_normalization_2/batchnorm/add_1:z:05model/transformer_block_1/dropout_3/Identity:output:0*
T0*+
_output_shapes
:?????????- 2!
model/transformer_block_1/add_1?
Nmodel/transformer_block_1/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
Nmodel/transformer_block_1/layer_normalization_3/moments/mean/reduction_indices?
<model/transformer_block_1/layer_normalization_3/moments/meanMean#model/transformer_block_1/add_1:z:0Wmodel/transformer_block_1/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2>
<model/transformer_block_1/layer_normalization_3/moments/mean?
Dmodel/transformer_block_1/layer_normalization_3/moments/StopGradientStopGradientEmodel/transformer_block_1/layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:?????????-2F
Dmodel/transformer_block_1/layer_normalization_3/moments/StopGradient?
Imodel/transformer_block_1/layer_normalization_3/moments/SquaredDifferenceSquaredDifference#model/transformer_block_1/add_1:z:0Mmodel/transformer_block_1/layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????- 2K
Imodel/transformer_block_1/layer_normalization_3/moments/SquaredDifference?
Rmodel/transformer_block_1/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2T
Rmodel/transformer_block_1/layer_normalization_3/moments/variance/reduction_indices?
@model/transformer_block_1/layer_normalization_3/moments/varianceMeanMmodel/transformer_block_1/layer_normalization_3/moments/SquaredDifference:z:0[model/transformer_block_1/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????-*
	keep_dims(2B
@model/transformer_block_1/layer_normalization_3/moments/variance?
?model/transformer_block_1/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52A
?model/transformer_block_1/layer_normalization_3/batchnorm/add/y?
=model/transformer_block_1/layer_normalization_3/batchnorm/addAddV2Imodel/transformer_block_1/layer_normalization_3/moments/variance:output:0Hmodel/transformer_block_1/layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????-2?
=model/transformer_block_1/layer_normalization_3/batchnorm/add?
?model/transformer_block_1/layer_normalization_3/batchnorm/RsqrtRsqrtAmodel/transformer_block_1/layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????-2A
?model/transformer_block_1/layer_normalization_3/batchnorm/Rsqrt?
Lmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpUmodel_transformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02N
Lmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp?
=model/transformer_block_1/layer_normalization_3/batchnorm/mulMulCmodel/transformer_block_1/layer_normalization_3/batchnorm/Rsqrt:y:0Tmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2?
=model/transformer_block_1/layer_normalization_3/batchnorm/mul?
?model/transformer_block_1/layer_normalization_3/batchnorm/mul_1Mul#model/transformer_block_1/add_1:z:0Amodel/transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2A
?model/transformer_block_1/layer_normalization_3/batchnorm/mul_1?
?model/transformer_block_1/layer_normalization_3/batchnorm/mul_2MulEmodel/transformer_block_1/layer_normalization_3/moments/mean:output:0Amodel/transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????- 2A
?model/transformer_block_1/layer_normalization_3/batchnorm/mul_2?
Hmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOpQmodel_transformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp?
=model/transformer_block_1/layer_normalization_3/batchnorm/subSubPmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp:value:0Cmodel/transformer_block_1/layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????- 2?
=model/transformer_block_1/layer_normalization_3/batchnorm/sub?
?model/transformer_block_1/layer_normalization_3/batchnorm/add_1AddV2Cmodel/transformer_block_1/layer_normalization_3/batchnorm/mul_1:z:0Amodel/transformer_block_1/layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????- 2A
?model/transformer_block_1/layer_normalization_3/batchnorm/add_1?
5model/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :27
5model/global_average_pooling1d/Mean/reduction_indices?
#model/global_average_pooling1d/MeanMeanCmodel/transformer_block_1/layer_normalization_3/batchnorm/add_1:z:0>model/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2%
#model/global_average_pooling1d/Mean?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2,model/global_average_pooling1d/Mean:output:0input_2&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????(2
model/concatenate/concat?
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource*
_output_shapes

:( *
dtype02%
#model/dense_4/MatMul/ReadVariableOp?
model/dense_4/MatMulMatMul!model/concatenate/concat:output:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model/dense_4/MatMul?
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/dense_4/BiasAdd/ReadVariableOp?
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model/dense_4/BiasAdd?
model/dense_4/ReluRelumodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
model/dense_4/Relu?
model/dropout_4/IdentityIdentity model/dense_4/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
model/dropout_4/Identity?
#model/dense_5/MatMul/ReadVariableOpReadVariableOp,model_dense_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#model/dense_5/MatMul/ReadVariableOp?
model/dense_5/MatMulMatMul!model/dropout_4/Identity:output:0+model/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model/dense_5/MatMul?
$model/dense_5/BiasAdd/ReadVariableOpReadVariableOp-model_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/dense_5/BiasAdd/ReadVariableOp?
model/dense_5/BiasAddBiasAddmodel/dense_5/MatMul:product:0,model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model/dense_5/BiasAdd?
model/dense_5/ReluRelumodel/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
model/dense_5/Relu?
model/dropout_5/IdentityIdentity model/dense_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
model/dropout_5/Identity?
#model/dense_6/MatMul/ReadVariableOpReadVariableOp,model_dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#model/dense_6/MatMul/ReadVariableOp?
model/dense_6/MatMulMatMul!model/dropout_5/Identity:output:0+model/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_6/MatMul?
$model/dense_6/BiasAdd/ReadVariableOpReadVariableOp-model_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_6/BiasAdd/ReadVariableOp?
model/dense_6/BiasAddBiasAddmodel/dense_6/MatMul:product:0,model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_6/BiasAdd?
IdentityIdentitymodel/dense_6/BiasAdd:output:0%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp%^model/dense_5/BiasAdd/ReadVariableOp$^model/dense_5/MatMul/ReadVariableOp%^model/dense_6/BiasAdd/ReadVariableOp$^model/dense_6/MatMul/ReadVariableOp>^model/token_and_position_embedding/embedding/embedding_lookup@^model/token_and_position_embedding/embedding_1/embedding_lookupI^model/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpM^model/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpI^model/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpM^model/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpU^model/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp_^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpH^model/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOpR^model/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpJ^model/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpT^model/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpJ^model/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpT^model/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpF^model/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpH^model/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpF^model/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpH^model/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????i:?????????::::::::::::::::::::::::2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp2L
$model/dense_5/BiasAdd/ReadVariableOp$model/dense_5/BiasAdd/ReadVariableOp2J
#model/dense_5/MatMul/ReadVariableOp#model/dense_5/MatMul/ReadVariableOp2L
$model/dense_6/BiasAdd/ReadVariableOp$model/dense_6/BiasAdd/ReadVariableOp2J
#model/dense_6/MatMul/ReadVariableOp#model/dense_6/MatMul/ReadVariableOp2~
=model/token_and_position_embedding/embedding/embedding_lookup=model/token_and_position_embedding/embedding/embedding_lookup2?
?model/token_and_position_embedding/embedding_1/embedding_lookup?model/token_and_position_embedding/embedding_1/embedding_lookup2?
Hmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpHmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp2?
Lmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpLmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp2?
Hmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpHmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp2?
Lmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpLmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp2?
Tmodel/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpTmodel/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp2?
^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2?
Gmodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOpGmodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOp2?
Qmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpQmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2?
Imodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpImodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOp2?
Smodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpSmodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2?
Imodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpImodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOp2?
Smodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpSmodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2?
Emodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpEmodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp2?
Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpGmodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp2?
Emodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpEmodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp2?
Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpGmodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:Q M
(
_output_shapes
:??????????i
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
W
+__inference_concatenate_layer_call_fn_90363
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_889372
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :?????????:Q M
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?H
?
G__inference_sequential_1_layer_call_and_return_conditional_losses_90590

inputs-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource-
)dense_3_tensordot_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity??dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp? dense_3/Tensordot/ReadVariableOp?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes?
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/freeh
dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape?
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2?
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod?
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1?
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack?
dense_2/Tensordot/transpose	Transposeinputs!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2
dense_2/Tensordot/transpose?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_2?
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2
dense_2/BiasAddt
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????- 2
dense_2/Relu?
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02"
 dense_3/Tensordot/ReadVariableOpz
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/axes?
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_3/Tensordot/free|
dense_3/Tensordot/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2
dense_3/Tensordot/Shape?
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/GatherV2/axis?
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2?
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axis?
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2_1|
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const?
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod?
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1?
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1?
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axis?
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat?
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stack?
dense_3/Tensordot/transpose	Transposedense_2/Relu:activations:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2
dense_3/Tensordot/transpose?
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_3/Tensordot/Reshape?
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_3/Tensordot/MatMul?
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_2?
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axis?
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1?
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2
dense_3/Tensordot?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2
dense_3/BiasAdd?
IdentityIdentitydense_3/BiasAdd:output:0^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????- ::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
?
|
'__inference_dense_2_layer_call_fn_90656

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_883372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????- ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
?
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_90395

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*

seed*2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_89047

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
,__inference_sequential_1_layer_call_fn_90603

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_884312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????- ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
?
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_90442

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype0*

seed*2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
B__inference_dense_5_layer_call_and_return_conditional_losses_89014

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
,__inference_sequential_1_layer_call_fn_88469
dense_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_2_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_884582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????- ::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????- 
'
_user_specified_namedense_2_input
?	
?
B__inference_dense_4_layer_call_and_return_conditional_losses_90374

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:( *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
? 
?
B__inference_dense_2_layer_call_and_return_conditional_losses_88337

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????- 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????- ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_89946
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_893342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????i:?????????::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????i
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?	
?
B__inference_dense_5_layer_call_and_return_conditional_losses_90421

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
|
'__inference_dense_4_layer_call_fn_90383

inputs
unknown
	unknown_0
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
B__inference_dense_4_layer_call_and_return_conditional_losses_889572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
M
1__inference_average_pooling1d_layer_call_fn_88302

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_882962
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
T
8__inference_global_average_pooling1d_layer_call_fn_90350

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_889232
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????- :S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
?
T
8__inference_global_average_pooling1d_layer_call_fn_90339

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_884852
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_1_layer_call_fn_88442
dense_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_2_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_884312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????- ::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????- 
'
_user_specified_namedense_2_input
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_88990

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
|
'__inference_dense_5_layer_call_fn_90430

inputs
unknown
	unknown_0
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
B__inference_dense_5_layer_call_and_return_conditional_losses_890142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
B__inference_dense_4_layer_call_and_return_conditional_losses_88957

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:( *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

?
3__inference_transformer_block_1_layer_call_fn_90291

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_886822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:?????????- ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
?
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_90345

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????- :S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
?5
?
@__inference_model_layer_call_and_return_conditional_losses_89150
input_1
input_2&
"token_and_position_embedding_89091&
"token_and_position_embedding_89093
transformer_block_1_89097
transformer_block_1_89099
transformer_block_1_89101
transformer_block_1_89103
transformer_block_1_89105
transformer_block_1_89107
transformer_block_1_89109
transformer_block_1_89111
transformer_block_1_89113
transformer_block_1_89115
transformer_block_1_89117
transformer_block_1_89119
transformer_block_1_89121
transformer_block_1_89123
transformer_block_1_89125
transformer_block_1_89127
dense_4_89132
dense_4_89134
dense_5_89138
dense_5_89140
dense_6_89144
dense_6_89146
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?4token_and_position_embedding/StatefulPartitionedCall?+transformer_block_1/StatefulPartitionedCall?
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1"token_and_position_embedding_89091"token_and_position_embedding_89093*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????i *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_8851726
4token_and_position_embedding/StatefulPartitionedCall?
!average_pooling1d/PartitionedCallPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_882962#
!average_pooling1d/PartitionedCall?
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0transformer_block_1_89097transformer_block_1_89099transformer_block_1_89101transformer_block_1_89103transformer_block_1_89105transformer_block_1_89107transformer_block_1_89109transformer_block_1_89111transformer_block_1_89113transformer_block_1_89115transformer_block_1_89117transformer_block_1_89119transformer_block_1_89121transformer_block_1_89123transformer_block_1_89125transformer_block_1_89127*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_888092-
+transformer_block_1/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall4transformer_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_889232*
(global_average_pooling1d/PartitionedCall?
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_889372
concatenate/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_89132dense_4_89134*
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
B__inference_dense_4_layer_call_and_return_conditional_losses_889572!
dense_4/StatefulPartitionedCall?
dropout_4/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_889902
dropout_4/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_5_89138dense_5_89140*
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
B__inference_dense_5_layer_call_and_return_conditional_losses_890142!
dense_5/StatefulPartitionedCall?
dropout_5/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_890472
dropout_5/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_6_89144dense_6_89146*
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
B__inference_dense_6_layer_call_and_return_conditional_losses_890702!
dense_6/StatefulPartitionedCall?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????i:?????????::::::::::::::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????i
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?	
?
B__inference_dense_6_layer_call_and_return_conditional_losses_90467

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?$
!__inference__traced_restore_91053
file_prefix#
assignvariableop_dense_4_kernel#
assignvariableop_1_dense_4_bias%
!assignvariableop_2_dense_5_kernel#
assignvariableop_3_dense_5_bias%
!assignvariableop_4_dense_6_kernel#
assignvariableop_5_dense_6_bias
assignvariableop_6_decay$
 assignvariableop_7_learning_rate
assignvariableop_8_momentum
assignvariableop_9_sgd_iterI
Eassignvariableop_10_token_and_position_embedding_embedding_embeddingsK
Gassignvariableop_11_token_and_position_embedding_embedding_1_embeddingsO
Kassignvariableop_12_transformer_block_1_multi_head_attention_1_query_kernelM
Iassignvariableop_13_transformer_block_1_multi_head_attention_1_query_biasM
Iassignvariableop_14_transformer_block_1_multi_head_attention_1_key_kernelK
Gassignvariableop_15_transformer_block_1_multi_head_attention_1_key_biasO
Kassignvariableop_16_transformer_block_1_multi_head_attention_1_value_kernelM
Iassignvariableop_17_transformer_block_1_multi_head_attention_1_value_biasZ
Vassignvariableop_18_transformer_block_1_multi_head_attention_1_attention_output_kernelX
Tassignvariableop_19_transformer_block_1_multi_head_attention_1_attention_output_bias&
"assignvariableop_20_dense_2_kernel$
 assignvariableop_21_dense_2_bias&
"assignvariableop_22_dense_3_kernel$
 assignvariableop_23_dense_3_biasG
Cassignvariableop_24_transformer_block_1_layer_normalization_2_gammaF
Bassignvariableop_25_transformer_block_1_layer_normalization_2_betaG
Cassignvariableop_26_transformer_block_1_layer_normalization_3_gammaF
Bassignvariableop_27_transformer_block_1_layer_normalization_3_beta
assignvariableop_28_total
assignvariableop_29_count3
/assignvariableop_30_sgd_dense_4_kernel_momentum1
-assignvariableop_31_sgd_dense_4_bias_momentum3
/assignvariableop_32_sgd_dense_5_kernel_momentum1
-assignvariableop_33_sgd_dense_5_bias_momentum3
/assignvariableop_34_sgd_dense_6_kernel_momentum1
-assignvariableop_35_sgd_dense_6_bias_momentumV
Rassignvariableop_36_sgd_token_and_position_embedding_embedding_embeddings_momentumX
Tassignvariableop_37_sgd_token_and_position_embedding_embedding_1_embeddings_momentum\
Xassignvariableop_38_sgd_transformer_block_1_multi_head_attention_1_query_kernel_momentumZ
Vassignvariableop_39_sgd_transformer_block_1_multi_head_attention_1_query_bias_momentumZ
Vassignvariableop_40_sgd_transformer_block_1_multi_head_attention_1_key_kernel_momentumX
Tassignvariableop_41_sgd_transformer_block_1_multi_head_attention_1_key_bias_momentum\
Xassignvariableop_42_sgd_transformer_block_1_multi_head_attention_1_value_kernel_momentumZ
Vassignvariableop_43_sgd_transformer_block_1_multi_head_attention_1_value_bias_momentumg
cassignvariableop_44_sgd_transformer_block_1_multi_head_attention_1_attention_output_kernel_momentume
aassignvariableop_45_sgd_transformer_block_1_multi_head_attention_1_attention_output_bias_momentum3
/assignvariableop_46_sgd_dense_2_kernel_momentum1
-assignvariableop_47_sgd_dense_2_bias_momentum3
/assignvariableop_48_sgd_dense_3_kernel_momentum1
-assignvariableop_49_sgd_dense_3_bias_momentumT
Passignvariableop_50_sgd_transformer_block_1_layer_normalization_2_gamma_momentumS
Oassignvariableop_51_sgd_transformer_block_1_layer_normalization_2_beta_momentumT
Passignvariableop_52_sgd_transformer_block_1_layer_normalization_3_gamma_momentumS
Oassignvariableop_53_sgd_transformer_block_1_layer_normalization_3_beta_momentum
identity_55??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
value?B?7B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/8/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_6_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_momentumIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_iterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpEassignvariableop_10_token_and_position_embedding_embedding_embeddingsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpGassignvariableop_11_token_and_position_embedding_embedding_1_embeddingsIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpKassignvariableop_12_transformer_block_1_multi_head_attention_1_query_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpIassignvariableop_13_transformer_block_1_multi_head_attention_1_query_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpIassignvariableop_14_transformer_block_1_multi_head_attention_1_key_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpGassignvariableop_15_transformer_block_1_multi_head_attention_1_key_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpKassignvariableop_16_transformer_block_1_multi_head_attention_1_value_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpIassignvariableop_17_transformer_block_1_multi_head_attention_1_value_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpVassignvariableop_18_transformer_block_1_multi_head_attention_1_attention_output_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpTassignvariableop_19_transformer_block_1_multi_head_attention_1_attention_output_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_2_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_2_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_3_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_3_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpCassignvariableop_24_transformer_block_1_layer_normalization_2_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpBassignvariableop_25_transformer_block_1_layer_normalization_2_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpCassignvariableop_26_transformer_block_1_layer_normalization_3_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpBassignvariableop_27_transformer_block_1_layer_normalization_3_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp/assignvariableop_30_sgd_dense_4_kernel_momentumIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp-assignvariableop_31_sgd_dense_4_bias_momentumIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp/assignvariableop_32_sgd_dense_5_kernel_momentumIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp-assignvariableop_33_sgd_dense_5_bias_momentumIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp/assignvariableop_34_sgd_dense_6_kernel_momentumIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp-assignvariableop_35_sgd_dense_6_bias_momentumIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpRassignvariableop_36_sgd_token_and_position_embedding_embedding_embeddings_momentumIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpTassignvariableop_37_sgd_token_and_position_embedding_embedding_1_embeddings_momentumIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpXassignvariableop_38_sgd_transformer_block_1_multi_head_attention_1_query_kernel_momentumIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpVassignvariableop_39_sgd_transformer_block_1_multi_head_attention_1_query_bias_momentumIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpVassignvariableop_40_sgd_transformer_block_1_multi_head_attention_1_key_kernel_momentumIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpTassignvariableop_41_sgd_transformer_block_1_multi_head_attention_1_key_bias_momentumIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpXassignvariableop_42_sgd_transformer_block_1_multi_head_attention_1_value_kernel_momentumIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpVassignvariableop_43_sgd_transformer_block_1_multi_head_attention_1_value_bias_momentumIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpcassignvariableop_44_sgd_transformer_block_1_multi_head_attention_1_attention_output_kernel_momentumIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpaassignvariableop_45_sgd_transformer_block_1_multi_head_attention_1_attention_output_bias_momentumIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp/assignvariableop_46_sgd_dense_2_kernel_momentumIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp-assignvariableop_47_sgd_dense_2_bias_momentumIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp/assignvariableop_48_sgd_dense_3_kernel_momentumIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp-assignvariableop_49_sgd_dense_3_bias_momentumIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpPassignvariableop_50_sgd_transformer_block_1_layer_normalization_2_gamma_momentumIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpOassignvariableop_51_sgd_transformer_block_1_layer_normalization_2_beta_momentumIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpPassignvariableop_52_sgd_transformer_block_1_layer_normalization_3_gamma_momentumIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpOassignvariableop_53_sgd_transformer_block_1_layer_normalization_3_beta_momentumIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_539
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_54?	
Identity_55IdentityIdentity_54:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_55"#
identity_55Identity_55:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_90447

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
E
)__inference_dropout_4_layer_call_fn_90410

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_889902
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?8
?
@__inference_model_layer_call_and_return_conditional_losses_89087
input_1
input_2&
"token_and_position_embedding_88528&
"token_and_position_embedding_88530
transformer_block_1_88885
transformer_block_1_88887
transformer_block_1_88889
transformer_block_1_88891
transformer_block_1_88893
transformer_block_1_88895
transformer_block_1_88897
transformer_block_1_88899
transformer_block_1_88901
transformer_block_1_88903
transformer_block_1_88905
transformer_block_1_88907
transformer_block_1_88909
transformer_block_1_88911
transformer_block_1_88913
transformer_block_1_88915
dense_4_88968
dense_4_88970
dense_5_89025
dense_5_89027
dense_6_89081
dense_6_89083
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?!dropout_5/StatefulPartitionedCall?4token_and_position_embedding/StatefulPartitionedCall?+transformer_block_1/StatefulPartitionedCall?
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1"token_and_position_embedding_88528"token_and_position_embedding_88530*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????i *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_8851726
4token_and_position_embedding/StatefulPartitionedCall?
!average_pooling1d/PartitionedCallPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_882962#
!average_pooling1d/PartitionedCall?
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0transformer_block_1_88885transformer_block_1_88887transformer_block_1_88889transformer_block_1_88891transformer_block_1_88893transformer_block_1_88895transformer_block_1_88897transformer_block_1_88899transformer_block_1_88901transformer_block_1_88903transformer_block_1_88905transformer_block_1_88907transformer_block_1_88909transformer_block_1_88911transformer_block_1_88913transformer_block_1_88915*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_886822-
+transformer_block_1/StatefulPartitionedCall?
(global_average_pooling1d/PartitionedCallPartitionedCall4transformer_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_889232*
(global_average_pooling1d/PartitionedCall?
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_889372
concatenate/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_88968dense_4_88970*
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
B__inference_dense_4_layer_call_and_return_conditional_losses_889572!
dense_4/StatefulPartitionedCall?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_889852#
!dropout_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_5_89025dense_5_89027*
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
B__inference_dense_5_layer_call_and_return_conditional_losses_890142!
dense_5/StatefulPartitionedCall?
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_890422#
!dropout_5/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_6_89081dense_6_89083*
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
B__inference_dense_6_layer_call_and_return_conditional_losses_890702!
dense_6/StatefulPartitionedCall?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????i:?????????::::::::::::::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????i
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?

?
3__inference_transformer_block_1_layer_call_fn_90328

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????- *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_888092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:?????????- ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
?
?
B__inference_dense_3_layer_call_and_return_conditional_losses_88383

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????- ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs
? 
?
B__inference_dense_2_layer_call_and_return_conditional_losses_90647

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????- 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????- 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????- 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????- 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????- 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????- ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????- 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_11
serving_default_input_1:0??????????i
;
input_20
serving_default_input_2:0?????????;
dense_60
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?)
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?%
_tf_keras_network?%{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 13500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d", "inbound_nodes": [[["token_and_position_embedding", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_1", "inbound_nodes": [[["average_pooling1d", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling1d", "inbound_nodes": [[["transformer_block_1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["global_average_pooling1d", 0, 0, {}], ["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dropout_4", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_5", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dropout_5", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["dense_6", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 13500]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 13500]}, {"class_name": "TensorShape", "items": [null, 8]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.004999999888241291, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 13500]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 13500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?
	token_emb
pos_emb
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "AveragePooling1D", "name": "average_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
att
ffn

layernorm1
 
layernorm2
!dropout1
"dropout2
#trainable_variables
$	variables
%regularization_losses
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TransformerBlock", "name": "transformer_block_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
'trainable_variables
(	variables
)regularization_losses
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GlobalAveragePooling1D", "name": "global_average_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?
+trainable_variables
,	variables
-regularization_losses
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32]}, {"class_name": "TensorShape", "items": [null, 8]}]}
?

/kernel
0bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
?
5trainable_variables
6	variables
7regularization_losses
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?

9kernel
:bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?

Ckernel
Dbias
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
	Idecay
Jlearning_rate
Kmomentum
Liter/momentum?0momentum?9momentum?:momentum?Cmomentum?Dmomentum?Mmomentum?Nmomentum?Omomentum?Pmomentum?Qmomentum?Rmomentum?Smomentum?Tmomentum?Umomentum?Vmomentum?Wmomentum?Xmomentum?Ymomentum?Zmomentum?[momentum?\momentum?]momentum?^momentum?"
	optimizer
?
M0
N1
O2
P3
Q4
R5
S6
T7
U8
V9
W10
X11
Y12
Z13
[14
\15
]16
^17
/18
019
920
:21
C22
D23"
trackable_list_wrapper
?
M0
N1
O2
P3
Q4
R5
S6
T7
U8
V9
W10
X11
Y12
Z13
[14
\15
]16
^17
/18
019
920
:21
C22
D23"
trackable_list_wrapper
 "
trackable_list_wrapper
?
_non_trainable_variables
`metrics
alayer_metrics
trainable_variables
blayer_regularization_losses

clayers
	variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
M
embeddings
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 5, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13500]}}
?
N
embeddings
htrainable_variables
i	variables
jregularization_losses
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13500, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
lnon_trainable_variables
mmetrics
nlayer_metrics
trainable_variables
olayer_regularization_losses

players
	variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
qnon_trainable_variables
rmetrics
slayer_metrics
trainable_variables
tlayer_regularization_losses

ulayers
	variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
v_query_dense
w
_key_dense
x_value_dense
y_softmax
z_dropout_layer
{_output_dense
|trainable_variables
}	variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MultiHeadAttention", "name": "multi_head_attention_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_1", "trainable": true, "dtype": "float32", "num_heads": 1, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
?
?layer_with_weights-0
?layer-0
?layer_with_weights-1
?layer-1
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 45, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_2_input"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 45, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_2_input"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
?
	?axis
	[gamma
\beta
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LayerNormalization", "name": "layer_normalization_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 32]}}
?
	?axis
	]gamma
^beta
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LayerNormalization", "name": "layer_normalization_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_3", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 32]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?
O0
P1
Q2
R3
S4
T5
U6
V7
W8
X9
Y10
Z11
[12
\13
]14
^15"
trackable_list_wrapper
?
O0
P1
Q2
R3
S4
T5
U6
V7
W8
X9
Y10
Z11
[12
\13
]14
^15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
#trainable_variables
 ?layer_regularization_losses
?layers
$	variables
%regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
'trainable_variables
 ?layer_regularization_losses
?layers
(	variables
)regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
+trainable_variables
 ?layer_regularization_losses
?layers
,	variables
-regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :( 2dense_4/kernel
: 2dense_4/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
1trainable_variables
 ?layer_regularization_losses
?layers
2	variables
3regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
5trainable_variables
 ?layer_regularization_losses
?layers
6	variables
7regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :  2dense_5/kernel
: 2dense_5/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
;trainable_variables
 ?layer_regularization_losses
?layers
<	variables
=regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
@	variables
Aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 : 2dense_6/kernel
:2dense_6/bias
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
Etrainable_variables
 ?layer_regularization_losses
?layers
F	variables
Gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
C:A 21token_and_position_embedding/embedding/embeddings
F:D	?i 23token_and_position_embedding/embedding_1/embeddings
M:K  27transformer_block_1/multi_head_attention_1/query/kernel
G:E 25transformer_block_1/multi_head_attention_1/query/bias
K:I  25transformer_block_1/multi_head_attention_1/key/kernel
E:C 23transformer_block_1/multi_head_attention_1/key/bias
M:K  27transformer_block_1/multi_head_attention_1/value/kernel
G:E 25transformer_block_1/multi_head_attention_1/value/bias
X:V  2Btransformer_block_1/multi_head_attention_1/attention_output/kernel
N:L 2@transformer_block_1/multi_head_attention_1/attention_output/bias
 :  2dense_2/kernel
: 2dense_2/bias
 :  2dense_3/kernel
: 2dense_3/bias
=:; 2/transformer_block_1/layer_normalization_2/gamma
<:: 2.transformer_block_1/layer_normalization_2/beta
=:; 2/transformer_block_1/layer_normalization_3/gamma
<:: 2.transformer_block_1/layer_normalization_3/beta
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
'
M0"
trackable_list_wrapper
'
M0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
dtrainable_variables
 ?layer_regularization_losses
?layers
e	variables
fregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
N0"
trackable_list_wrapper
'
N0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
htrainable_variables
 ?layer_regularization_losses
?layers
i	variables
jregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?partial_output_shape
?full_output_shape

Okernel
Pbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 32]}}
?
?partial_output_shape
?full_output_shape

Qkernel
Rbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 32]}}
?
?partial_output_shape
?full_output_shape

Skernel
Tbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 32]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
?
?partial_output_shape
?full_output_shape

Ukernel
Vbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 32], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 1, 32]}}
X
O0
P1
Q2
R3
S4
T5
U6
V7"
trackable_list_wrapper
X
O0
P1
Q2
R3
S4
T5
U6
V7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
|trainable_variables
 ?layer_regularization_losses
?layers
}	variables
~regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Wkernel
Xbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 32]}}
?

Ykernel
Zbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 32]}}
<
W0
X1
Y2
Z3"
trackable_list_wrapper
<
W0
X1
Y2
Z3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
0
1
2
 3
!4
"5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
v0
w1
x2
y3
z4
{5"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
?layer_metrics
?trainable_variables
 ?layer_regularization_losses
?layers
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
+:)( 2SGD/dense_4/kernel/momentum
%:# 2SGD/dense_4/bias/momentum
+:)  2SGD/dense_5/kernel/momentum
%:# 2SGD/dense_5/bias/momentum
+:) 2SGD/dense_6/kernel/momentum
%:#2SGD/dense_6/bias/momentum
N:L 2>SGD/token_and_position_embedding/embedding/embeddings/momentum
Q:O	?i 2@SGD/token_and_position_embedding/embedding_1/embeddings/momentum
X:V  2DSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentum
R:P 2BSGD/transformer_block_1/multi_head_attention_1/query/bias/momentum
V:T  2BSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentum
P:N 2@SGD/transformer_block_1/multi_head_attention_1/key/bias/momentum
X:V  2DSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentum
R:P 2BSGD/transformer_block_1/multi_head_attention_1/value/bias/momentum
c:a  2OSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentum
Y:W 2MSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentum
+:)  2SGD/dense_2/kernel/momentum
%:# 2SGD/dense_2/bias/momentum
+:)  2SGD/dense_3/kernel/momentum
%:# 2SGD/dense_3/bias/momentum
H:F 2<SGD/transformer_block_1/layer_normalization_2/gamma/momentum
G:E 2;SGD/transformer_block_1/layer_normalization_2/beta/momentum
H:F 2<SGD/transformer_block_1/layer_normalization_3/gamma/momentum
G:E 2;SGD/transformer_block_1/layer_normalization_3/beta/momentum
?2?
 __inference__wrapped_model_88287?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *O?L
J?G
"?
input_1??????????i
!?
input_2?????????
?2?
%__inference_model_layer_call_fn_89946
%__inference_model_layer_call_fn_89385
%__inference_model_layer_call_fn_89892
%__inference_model_layer_call_fn_89268?
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
?2?
@__inference_model_layer_call_and_return_conditional_losses_89660
@__inference_model_layer_call_and_return_conditional_losses_89838
@__inference_model_layer_call_and_return_conditional_losses_89087
@__inference_model_layer_call_and_return_conditional_losses_89150?
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
?2?
<__inference_token_and_position_embedding_layer_call_fn_89979?
???
FullArgSpec
args?
jself
jx
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
?2?
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_89970?
???
FullArgSpec
args?
jself
jx
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
?2?
1__inference_average_pooling1d_layer_call_fn_88302?
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
annotations? *3?0
.?+'???????????????????????????
?2?
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_88296?
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
annotations? *3?0
.?+'???????????????????????????
?2?
3__inference_transformer_block_1_layer_call_fn_90328
3__inference_transformer_block_1_layer_call_fn_90291?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_90254
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_90127?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
8__inference_global_average_pooling1d_layer_call_fn_90339
8__inference_global_average_pooling1d_layer_call_fn_90350?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_90345
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_90334?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_concatenate_layer_call_fn_90363?
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
?2?
F__inference_concatenate_layer_call_and_return_conditional_losses_90357?
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
?2?
'__inference_dense_4_layer_call_fn_90383?
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
?2?
B__inference_dense_4_layer_call_and_return_conditional_losses_90374?
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
?2?
)__inference_dropout_4_layer_call_fn_90405
)__inference_dropout_4_layer_call_fn_90410?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_4_layer_call_and_return_conditional_losses_90395
D__inference_dropout_4_layer_call_and_return_conditional_losses_90400?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_5_layer_call_fn_90430?
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
?2?
B__inference_dense_5_layer_call_and_return_conditional_losses_90421?
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
?2?
)__inference_dropout_5_layer_call_fn_90452
)__inference_dropout_5_layer_call_fn_90457?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_5_layer_call_and_return_conditional_losses_90447
D__inference_dropout_5_layer_call_and_return_conditional_losses_90442?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_6_layer_call_fn_90476?
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
?2?
B__inference_dense_6_layer_call_and_return_conditional_losses_90467?
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
#__inference_signature_wrapper_89447input_1input_2"?
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_sequential_1_layer_call_fn_88442
,__inference_sequential_1_layer_call_fn_90616
,__inference_sequential_1_layer_call_fn_90603
,__inference_sequential_1_layer_call_fn_88469?
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
?2?
G__inference_sequential_1_layer_call_and_return_conditional_losses_90533
G__inference_sequential_1_layer_call_and_return_conditional_losses_88400
G__inference_sequential_1_layer_call_and_return_conditional_losses_90590
G__inference_sequential_1_layer_call_and_return_conditional_losses_88414?
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
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
?2??
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
?2?
'__inference_dense_2_layer_call_fn_90656?
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
?2?
B__inference_dense_2_layer_call_and_return_conditional_losses_90647?
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
?2?
'__inference_dense_3_layer_call_fn_90695?
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
?2?
B__inference_dense_3_layer_call_and_return_conditional_losses_90686?
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
 ?
 __inference__wrapped_model_88287?NMOPQRSTUV[\WXYZ]^/09:CDY?V
O?L
J?G
"?
input_1??????????i
!?
input_2?????????
? "1?.
,
dense_6!?
dense_6??????????
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_88296?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
1__inference_average_pooling1d_layer_call_fn_88302wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
F__inference_concatenate_layer_call_and_return_conditional_losses_90357?Z?W
P?M
K?H
"?
inputs/0????????? 
"?
inputs/1?????????
? "%?"
?
0?????????(
? ?
+__inference_concatenate_layer_call_fn_90363vZ?W
P?M
K?H
"?
inputs/0????????? 
"?
inputs/1?????????
? "??????????(?
B__inference_dense_2_layer_call_and_return_conditional_losses_90647dWX3?0
)?&
$?!
inputs?????????- 
? ")?&
?
0?????????- 
? ?
'__inference_dense_2_layer_call_fn_90656WWX3?0
)?&
$?!
inputs?????????- 
? "??????????- ?
B__inference_dense_3_layer_call_and_return_conditional_losses_90686dYZ3?0
)?&
$?!
inputs?????????- 
? ")?&
?
0?????????- 
? ?
'__inference_dense_3_layer_call_fn_90695WYZ3?0
)?&
$?!
inputs?????????- 
? "??????????- ?
B__inference_dense_4_layer_call_and_return_conditional_losses_90374\/0/?,
%?"
 ?
inputs?????????(
? "%?"
?
0????????? 
? z
'__inference_dense_4_layer_call_fn_90383O/0/?,
%?"
 ?
inputs?????????(
? "?????????? ?
B__inference_dense_5_layer_call_and_return_conditional_losses_90421\9:/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? z
'__inference_dense_5_layer_call_fn_90430O9:/?,
%?"
 ?
inputs????????? 
? "?????????? ?
B__inference_dense_6_layer_call_and_return_conditional_losses_90467\CD/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? z
'__inference_dense_6_layer_call_fn_90476OCD/?,
%?"
 ?
inputs????????? 
? "???????????
D__inference_dropout_4_layer_call_and_return_conditional_losses_90395\3?0
)?&
 ?
inputs????????? 
p
? "%?"
?
0????????? 
? ?
D__inference_dropout_4_layer_call_and_return_conditional_losses_90400\3?0
)?&
 ?
inputs????????? 
p 
? "%?"
?
0????????? 
? |
)__inference_dropout_4_layer_call_fn_90405O3?0
)?&
 ?
inputs????????? 
p
? "?????????? |
)__inference_dropout_4_layer_call_fn_90410O3?0
)?&
 ?
inputs????????? 
p 
? "?????????? ?
D__inference_dropout_5_layer_call_and_return_conditional_losses_90442\3?0
)?&
 ?
inputs????????? 
p
? "%?"
?
0????????? 
? ?
D__inference_dropout_5_layer_call_and_return_conditional_losses_90447\3?0
)?&
 ?
inputs????????? 
p 
? "%?"
?
0????????? 
? |
)__inference_dropout_5_layer_call_fn_90452O3?0
)?&
 ?
inputs????????? 
p
? "?????????? |
)__inference_dropout_5_layer_call_fn_90457O3?0
)?&
 ?
inputs????????? 
p 
? "?????????? ?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_90334{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_90345`7?4
-?*
$?!
inputs?????????- 

 
? "%?"
?
0????????? 
? ?
8__inference_global_average_pooling1d_layer_call_fn_90339nI?F
??<
6?3
inputs'???????????????????????????

 
? "!????????????????????
8__inference_global_average_pooling1d_layer_call_fn_90350S7?4
-?*
$?!
inputs?????????- 

 
? "?????????? ?
@__inference_model_layer_call_and_return_conditional_losses_89087?NMOPQRSTUV[\WXYZ]^/09:CDa?^
W?T
J?G
"?
input_1??????????i
!?
input_2?????????
p

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_89150?NMOPQRSTUV[\WXYZ]^/09:CDa?^
W?T
J?G
"?
input_1??????????i
!?
input_2?????????
p 

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_89660?NMOPQRSTUV[\WXYZ]^/09:CDc?`
Y?V
L?I
#? 
inputs/0??????????i
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_89838?NMOPQRSTUV[\WXYZ]^/09:CDc?`
Y?V
L?I
#? 
inputs/0??????????i
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
%__inference_model_layer_call_fn_89268?NMOPQRSTUV[\WXYZ]^/09:CDa?^
W?T
J?G
"?
input_1??????????i
!?
input_2?????????
p

 
? "???????????
%__inference_model_layer_call_fn_89385?NMOPQRSTUV[\WXYZ]^/09:CDa?^
W?T
J?G
"?
input_1??????????i
!?
input_2?????????
p 

 
? "???????????
%__inference_model_layer_call_fn_89892?NMOPQRSTUV[\WXYZ]^/09:CDc?`
Y?V
L?I
#? 
inputs/0??????????i
"?
inputs/1?????????
p

 
? "???????????
%__inference_model_layer_call_fn_89946?NMOPQRSTUV[\WXYZ]^/09:CDc?`
Y?V
L?I
#? 
inputs/0??????????i
"?
inputs/1?????????
p 

 
? "???????????
G__inference_sequential_1_layer_call_and_return_conditional_losses_88400uWXYZB??
8?5
+?(
dense_2_input?????????- 
p

 
? ")?&
?
0?????????- 
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_88414uWXYZB??
8?5
+?(
dense_2_input?????????- 
p 

 
? ")?&
?
0?????????- 
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_90533nWXYZ;?8
1?.
$?!
inputs?????????- 
p

 
? ")?&
?
0?????????- 
? ?
G__inference_sequential_1_layer_call_and_return_conditional_losses_90590nWXYZ;?8
1?.
$?!
inputs?????????- 
p 

 
? ")?&
?
0?????????- 
? ?
,__inference_sequential_1_layer_call_fn_88442hWXYZB??
8?5
+?(
dense_2_input?????????- 
p

 
? "??????????- ?
,__inference_sequential_1_layer_call_fn_88469hWXYZB??
8?5
+?(
dense_2_input?????????- 
p 

 
? "??????????- ?
,__inference_sequential_1_layer_call_fn_90603aWXYZ;?8
1?.
$?!
inputs?????????- 
p

 
? "??????????- ?
,__inference_sequential_1_layer_call_fn_90616aWXYZ;?8
1?.
$?!
inputs?????????- 
p 

 
? "??????????- ?
#__inference_signature_wrapper_89447?NMOPQRSTUV[\WXYZ]^/09:CDj?g
? 
`?]
-
input_1"?
input_1??????????i
,
input_2!?
input_2?????????"1?.
,
dense_6!?
dense_6??????????
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_89970]NM+?(
!?
?
x??????????i
? "*?'
 ?
0??????????i 
? ?
<__inference_token_and_position_embedding_layer_call_fn_89979PNM+?(
!?
?
x??????????i
? "???????????i ?
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_90127vOPQRSTUV[\WXYZ]^7?4
-?*
$?!
inputs?????????- 
p
? ")?&
?
0?????????- 
? ?
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_90254vOPQRSTUV[\WXYZ]^7?4
-?*
$?!
inputs?????????- 
p 
? ")?&
?
0?????????- 
? ?
3__inference_transformer_block_1_layer_call_fn_90291iOPQRSTUV[\WXYZ]^7?4
-?*
$?!
inputs?????????- 
p
? "??????????- ?
3__inference_transformer_block_1_layer_call_fn_90328iOPQRSTUV[\WXYZ]^7?4
-?*
$?!
inputs?????????- 
p 
? "??????????- 