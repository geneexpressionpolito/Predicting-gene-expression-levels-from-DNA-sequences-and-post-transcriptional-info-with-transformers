µÔ/
¿!!
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
¼
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

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

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
­
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

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
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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

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
dtypetype
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
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
list(type)(0
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

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
¾
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
executor_typestring 
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
ö
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.1-0-g85c8b2a817f8âµ)
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:  *
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
: *
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  * 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:	  *
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
: *
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
: *
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
: *
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
¢
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	è@*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	è@*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:@*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:@@*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:@*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:@*
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
¾
1token_and_position_embedding/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *B
shared_name31token_and_position_embedding/embedding/embeddings
·
Etoken_and_position_embedding/embedding/embeddings/Read/ReadVariableOpReadVariableOp1token_and_position_embedding/embedding/embeddings*
_output_shapes

: *
dtype0
Ã
3token_and_position_embedding/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	R *D
shared_name53token_and_position_embedding/embedding_1/embeddings
¼
Gtoken_and_position_embedding/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp3token_and_position_embedding/embedding_1/embeddings*
_output_shapes
:	R *
dtype0
Î
7transformer_block_1/multi_head_attention_1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_1/multi_head_attention_1/query/kernel
Ç
Ktransformer_block_1/multi_head_attention_1/query/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_1/multi_head_attention_1/query/kernel*"
_output_shapes
:  *
dtype0
Æ
5transformer_block_1/multi_head_attention_1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_1/multi_head_attention_1/query/bias
¿
Itransformer_block_1/multi_head_attention_1/query/bias/Read/ReadVariableOpReadVariableOp5transformer_block_1/multi_head_attention_1/query/bias*
_output_shapes

: *
dtype0
Ê
5transformer_block_1/multi_head_attention_1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *F
shared_name75transformer_block_1/multi_head_attention_1/key/kernel
Ã
Itransformer_block_1/multi_head_attention_1/key/kernel/Read/ReadVariableOpReadVariableOp5transformer_block_1/multi_head_attention_1/key/kernel*"
_output_shapes
:  *
dtype0
Â
3transformer_block_1/multi_head_attention_1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *D
shared_name53transformer_block_1/multi_head_attention_1/key/bias
»
Gtransformer_block_1/multi_head_attention_1/key/bias/Read/ReadVariableOpReadVariableOp3transformer_block_1/multi_head_attention_1/key/bias*
_output_shapes

: *
dtype0
Î
7transformer_block_1/multi_head_attention_1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_1/multi_head_attention_1/value/kernel
Ç
Ktransformer_block_1/multi_head_attention_1/value/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_1/multi_head_attention_1/value/kernel*"
_output_shapes
:  *
dtype0
Æ
5transformer_block_1/multi_head_attention_1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_1/multi_head_attention_1/value/bias
¿
Itransformer_block_1/multi_head_attention_1/value/bias/Read/ReadVariableOpReadVariableOp5transformer_block_1/multi_head_attention_1/value/bias*
_output_shapes

: *
dtype0
ä
Btransformer_block_1/multi_head_attention_1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBtransformer_block_1/multi_head_attention_1/attention_output/kernel
Ý
Vtransformer_block_1/multi_head_attention_1/attention_output/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_1/multi_head_attention_1/attention_output/kernel*"
_output_shapes
:  *
dtype0
Ø
@transformer_block_1/multi_head_attention_1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_1/multi_head_attention_1/attention_output/bias
Ñ
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
: @*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

: @*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:@*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:@ *
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
¶
/transformer_block_1/layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_1/layer_normalization_2/gamma
¯
Ctransformer_block_1/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_1/layer_normalization_2/gamma*
_output_shapes
: *
dtype0
´
.transformer_block_1/layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_1/layer_normalization_2/beta
­
Btransformer_block_1/layer_normalization_2/beta/Read/ReadVariableOpReadVariableOp.transformer_block_1/layer_normalization_2/beta*
_output_shapes
: *
dtype0
¶
/transformer_block_1/layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_1/layer_normalization_3/gamma
¯
Ctransformer_block_1/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_1/layer_normalization_3/gamma*
_output_shapes
: *
dtype0
´
.transformer_block_1/layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_1/layer_normalization_3/beta
­
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

SGD/conv1d/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *+
shared_nameSGD/conv1d/kernel/momentum

.SGD/conv1d/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d/kernel/momentum*"
_output_shapes
:  *
dtype0

SGD/conv1d/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameSGD/conv1d/bias/momentum

,SGD/conv1d/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d/bias/momentum*
_output_shapes
: *
dtype0

SGD/conv1d_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *-
shared_nameSGD/conv1d_1/kernel/momentum

0SGD/conv1d_1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_1/kernel/momentum*"
_output_shapes
:	  *
dtype0

SGD/conv1d_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/conv1d_1/bias/momentum

.SGD/conv1d_1/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_1/bias/momentum*
_output_shapes
: *
dtype0
¤
&SGD/batch_normalization/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&SGD/batch_normalization/gamma/momentum

:SGD/batch_normalization/gamma/momentum/Read/ReadVariableOpReadVariableOp&SGD/batch_normalization/gamma/momentum*
_output_shapes
: *
dtype0
¢
%SGD/batch_normalization/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%SGD/batch_normalization/beta/momentum

9SGD/batch_normalization/beta/momentum/Read/ReadVariableOpReadVariableOp%SGD/batch_normalization/beta/momentum*
_output_shapes
: *
dtype0
¨
(SGD/batch_normalization_1/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(SGD/batch_normalization_1/gamma/momentum
¡
<SGD/batch_normalization_1/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_1/gamma/momentum*
_output_shapes
: *
dtype0
¦
'SGD/batch_normalization_1/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'SGD/batch_normalization_1/beta/momentum

;SGD/batch_normalization_1/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_1/beta/momentum*
_output_shapes
: *
dtype0

SGD/dense_4/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	è@*,
shared_nameSGD/dense_4/kernel/momentum

/SGD/dense_4/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_4/kernel/momentum*
_output_shapes
:	è@*
dtype0

SGD/dense_4/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameSGD/dense_4/bias/momentum

-SGD/dense_4/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_4/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_5/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*,
shared_nameSGD/dense_5/kernel/momentum

/SGD/dense_5/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_5/kernel/momentum*
_output_shapes

:@@*
dtype0

SGD/dense_5/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameSGD/dense_5/bias/momentum

-SGD/dense_5/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_5/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_6/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*,
shared_nameSGD/dense_6/kernel/momentum

/SGD/dense_6/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_6/kernel/momentum*
_output_shapes

:@*
dtype0

SGD/dense_6/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameSGD/dense_6/bias/momentum

-SGD/dense_6/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_6/bias/momentum*
_output_shapes
:*
dtype0
Ø
>SGD/token_and_position_embedding/embedding/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *O
shared_name@>SGD/token_and_position_embedding/embedding/embeddings/momentum
Ñ
RSGD/token_and_position_embedding/embedding/embeddings/momentum/Read/ReadVariableOpReadVariableOp>SGD/token_and_position_embedding/embedding/embeddings/momentum*
_output_shapes

: *
dtype0
Ý
@SGD/token_and_position_embedding/embedding_1/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	R *Q
shared_nameB@SGD/token_and_position_embedding/embedding_1/embeddings/momentum
Ö
TSGD/token_and_position_embedding/embedding_1/embeddings/momentum/Read/ReadVariableOpReadVariableOp@SGD/token_and_position_embedding/embedding_1/embeddings/momentum*
_output_shapes
:	R *
dtype0
è
DSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentum
á
XSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentum*"
_output_shapes
:  *
dtype0
à
BSGD/transformer_block_1/multi_head_attention_1/query/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_1/multi_head_attention_1/query/bias/momentum
Ù
VSGD/transformer_block_1/multi_head_attention_1/query/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_1/multi_head_attention_1/query/bias/momentum*
_output_shapes

: *
dtype0
ä
BSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentum
Ý
VSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentum*"
_output_shapes
:  *
dtype0
Ü
@SGD/transformer_block_1/multi_head_attention_1/key/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *Q
shared_nameB@SGD/transformer_block_1/multi_head_attention_1/key/bias/momentum
Õ
TSGD/transformer_block_1/multi_head_attention_1/key/bias/momentum/Read/ReadVariableOpReadVariableOp@SGD/transformer_block_1/multi_head_attention_1/key/bias/momentum*
_output_shapes

: *
dtype0
è
DSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentum
á
XSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentum*"
_output_shapes
:  *
dtype0
à
BSGD/transformer_block_1/multi_head_attention_1/value/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_1/multi_head_attention_1/value/bias/momentum
Ù
VSGD/transformer_block_1/multi_head_attention_1/value/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_1/multi_head_attention_1/value/bias/momentum*
_output_shapes

: *
dtype0
þ
OSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *`
shared_nameQOSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentum
÷
cSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentum/Read/ReadVariableOpReadVariableOpOSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentum*"
_output_shapes
:  *
dtype0
ò
MSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *^
shared_nameOMSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentum
ë
aSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentum/Read/ReadVariableOpReadVariableOpMSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentum*
_output_shapes
: *
dtype0

SGD/dense_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*,
shared_nameSGD/dense_2/kernel/momentum

/SGD/dense_2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_2/kernel/momentum*
_output_shapes

: @*
dtype0

SGD/dense_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameSGD/dense_2/bias/momentum

-SGD/dense_2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_2/bias/momentum*
_output_shapes
:@*
dtype0

SGD/dense_3/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *,
shared_nameSGD/dense_3/kernel/momentum

/SGD/dense_3/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_3/kernel/momentum*
_output_shapes

:@ *
dtype0

SGD/dense_3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameSGD/dense_3/bias/momentum

-SGD/dense_3/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_3/bias/momentum*
_output_shapes
: *
dtype0
Ð
<SGD/transformer_block_1/layer_normalization_2/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_1/layer_normalization_2/gamma/momentum
É
PSGD/transformer_block_1/layer_normalization_2/gamma/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_1/layer_normalization_2/gamma/momentum*
_output_shapes
: *
dtype0
Î
;SGD/transformer_block_1/layer_normalization_2/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;SGD/transformer_block_1/layer_normalization_2/beta/momentum
Ç
OSGD/transformer_block_1/layer_normalization_2/beta/momentum/Read/ReadVariableOpReadVariableOp;SGD/transformer_block_1/layer_normalization_2/beta/momentum*
_output_shapes
: *
dtype0
Ð
<SGD/transformer_block_1/layer_normalization_3/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_1/layer_normalization_3/gamma/momentum
É
PSGD/transformer_block_1/layer_normalization_3/gamma/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_1/layer_normalization_3/gamma/momentum*
_output_shapes
: *
dtype0
Î
;SGD/transformer_block_1/layer_normalization_3/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;SGD/transformer_block_1/layer_normalization_3/beta/momentum
Ç
OSGD/transformer_block_1/layer_normalization_3/beta/momentum/Read/ReadVariableOpReadVariableOp;SGD/transformer_block_1/layer_normalization_3/beta/momentum*
_output_shapes
: *
dtype0

NoOpNoOp
÷²
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*±²
value¦²B¢² B²
Û
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
n
	token_emb
pos_emb
regularization_losses
	variables
trainable_variables
	keras_api
h

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
R
&regularization_losses
'	variables
(trainable_variables
)	keras_api
h

*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
R
0regularization_losses
1	variables
2trainable_variables
3	keras_api
R
4regularization_losses
5	variables
6trainable_variables
7	keras_api

8axis
	9gamma
:beta
;moving_mean
<moving_variance
=regularization_losses
>	variables
?trainable_variables
@	keras_api

Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
R
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
 
Natt
Offn
P
layernorm1
Q
layernorm2
Rdropout1
Sdropout2
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
R
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
 
R
\regularization_losses
]	variables
^trainable_variables
_	keras_api
h

`kernel
abias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
R
fregularization_losses
g	variables
htrainable_variables
i	keras_api
h

jkernel
kbias
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
R
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
h

tkernel
ubias
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
æ
	zdecay
{learning_rate
|momentum
}iter momentum!momentum*momentum+momentum9momentum:momentumBmomentumCmomentum`momentumamomentumjmomentumkmomentumtmomentumumomentum~momentummomentum momentum¡momentum¢momentum£momentum¤momentum¥momentum¦momentum§momentum¨momentum©momentumªmomentum«momentum¬momentum­momentum®momentum¯momentum°
 
¦
~0
1
 2
!3
*4
+5
96
:7
;8
<9
B10
C11
D12
E13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
`30
a31
j32
k33
t34
u35

~0
1
 2
!3
*4
+5
96
:7
B8
C9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
`26
a27
j28
k29
t30
u31
²
regularization_losses
metrics
	variables
trainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
 
f
~
embeddings
regularization_losses
	variables
trainable_variables
	keras_api
f

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
 

~0
1

~0
1
²
regularization_losses
metrics
	variables
trainable_variables
layer_metrics
 layer_regularization_losses
 non_trainable_variables
¡layers
YW
VARIABLE_VALUEconv1d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
²
"regularization_losses
¢metrics
#	variables
$trainable_variables
£layer_metrics
 ¤layer_regularization_losses
¥non_trainable_variables
¦layers
 
 
 
²
&regularization_losses
§metrics
'	variables
(trainable_variables
¨layer_metrics
 ©layer_regularization_losses
ªnon_trainable_variables
«layers
[Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1

*0
+1
²
,regularization_losses
¬metrics
-	variables
.trainable_variables
­layer_metrics
 ®layer_regularization_losses
¯non_trainable_variables
°layers
 
 
 
²
0regularization_losses
±metrics
1	variables
2trainable_variables
²layer_metrics
 ³layer_regularization_losses
´non_trainable_variables
µlayers
 
 
 
²
4regularization_losses
¶metrics
5	variables
6trainable_variables
·layer_metrics
 ¸layer_regularization_losses
¹non_trainable_variables
ºlayers
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1
;2
<3

90
:1
²
=regularization_losses
»metrics
>	variables
?trainable_variables
¼layer_metrics
 ½layer_regularization_losses
¾non_trainable_variables
¿layers
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

B0
C1
D2
E3

B0
C1
²
Fregularization_losses
Àmetrics
G	variables
Htrainable_variables
Álayer_metrics
 Âlayer_regularization_losses
Ãnon_trainable_variables
Älayers
 
 
 
²
Jregularization_losses
Åmetrics
K	variables
Ltrainable_variables
Ælayer_metrics
 Çlayer_regularization_losses
Ènon_trainable_variables
Élayers
Å
Ê_query_dense
Ë
_key_dense
Ì_value_dense
Í_softmax
Î_dropout_layer
Ï_output_dense
Ðregularization_losses
Ñ	variables
Òtrainable_variables
Ó	keras_api
¨
Ôlayer_with_weights-0
Ôlayer-0
Õlayer_with_weights-1
Õlayer-1
Öregularization_losses
×	variables
Øtrainable_variables
Ù	keras_api
x
	Úaxis

gamma
	beta
Ûregularization_losses
Ü	variables
Ýtrainable_variables
Þ	keras_api
x
	ßaxis

gamma
	beta
àregularization_losses
á	variables
âtrainable_variables
ã	keras_api
V
äregularization_losses
å	variables
ætrainable_variables
ç	keras_api
V
èregularization_losses
é	variables
êtrainable_variables
ë	keras_api
 

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
²
Tregularization_losses
ìmetrics
U	variables
Vtrainable_variables
ílayer_metrics
 îlayer_regularization_losses
ïnon_trainable_variables
ðlayers
 
 
 
²
Xregularization_losses
ñmetrics
Y	variables
Ztrainable_variables
òlayer_metrics
 ólayer_regularization_losses
ônon_trainable_variables
õlayers
 
 
 
²
\regularization_losses
ömetrics
]	variables
^trainable_variables
÷layer_metrics
 ølayer_regularization_losses
ùnon_trainable_variables
úlayers
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

`0
a1

`0
a1
²
bregularization_losses
ûmetrics
c	variables
dtrainable_variables
ülayer_metrics
 ýlayer_regularization_losses
þnon_trainable_variables
ÿlayers
 
 
 
²
fregularization_losses
metrics
g	variables
htrainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

j0
k1

j0
k1
²
lregularization_losses
metrics
m	variables
ntrainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
 
 
 
²
pregularization_losses
metrics
q	variables
rtrainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

t0
u1

t0
u1
²
vregularization_losses
metrics
w	variables
xtrainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE1token_and_position_embedding/embedding/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE3token_and_position_embedding/embedding_1/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7transformer_block_1/multi_head_attention_1/query/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_1/multi_head_attention_1/query/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_1/multi_head_attention_1/key/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE3transformer_block_1/multi_head_attention_1/key/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7transformer_block_1/multi_head_attention_1/value/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_1/multi_head_attention_1/value/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEBtransformer_block_1/multi_head_attention_1/attention_output/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE@transformer_block_1/multi_head_attention_1/attention_output/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_2/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_2/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_3/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_3/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_1/layer_normalization_2/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.transformer_block_1/layer_normalization_2/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_1/layer_normalization_3/gamma'variables/28/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.transformer_block_1/layer_normalization_3/beta'variables/29/.ATTRIBUTES/VARIABLE_VALUE

0
 
 

;0
<1
D2
E3

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
12
13
14
15
16
17
18
 

~0

~0
µ
regularization_losses
metrics
	variables
trainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
 

0

0
µ
regularization_losses
metrics
	variables
trainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
 
 
 
 

0
1
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

;0
<1
 
 
 
 

D0
E1
 
 
 
 
 
 
¡
partial_output_shape
 full_output_shape
kernel
	bias
¡regularization_losses
¢	variables
£trainable_variables
¤	keras_api
¡
¥partial_output_shape
¦full_output_shape
kernel
	bias
§regularization_losses
¨	variables
©trainable_variables
ª	keras_api
¡
«partial_output_shape
¬full_output_shape
kernel
	bias
­regularization_losses
®	variables
¯trainable_variables
°	keras_api
V
±regularization_losses
²	variables
³trainable_variables
´	keras_api
V
µregularization_losses
¶	variables
·trainable_variables
¸	keras_api
¡
¹partial_output_shape
ºfull_output_shape
kernel
	bias
»regularization_losses
¼	variables
½trainable_variables
¾	keras_api
 
@
0
1
2
3
4
5
6
7
@
0
1
2
3
4
5
6
7
µ
Ðregularization_losses
¿metrics
Ñ	variables
Òtrainable_variables
Àlayer_metrics
 Álayer_regularization_losses
Ânon_trainable_variables
Ãlayers
n
kernel
	bias
Äregularization_losses
Å	variables
Ætrainable_variables
Ç	keras_api
n
kernel
	bias
Èregularization_losses
É	variables
Êtrainable_variables
Ë	keras_api
 
 
0
1
2
3
 
0
1
2
3
µ
Öregularization_losses
Ìmetrics
×	variables
Øtrainable_variables
Ílayer_metrics
 Îlayer_regularization_losses
Ïnon_trainable_variables
Ðlayers
 
 

0
1

0
1
µ
Ûregularization_losses
Ñmetrics
Ü	variables
Ýtrainable_variables
Òlayer_metrics
 Ólayer_regularization_losses
Ônon_trainable_variables
Õlayers
 
 

0
1

0
1
µ
àregularization_losses
Ömetrics
á	variables
âtrainable_variables
×layer_metrics
 Ølayer_regularization_losses
Ùnon_trainable_variables
Úlayers
 
 
 
µ
äregularization_losses
Ûmetrics
å	variables
ætrainable_variables
Ülayer_metrics
 Ýlayer_regularization_losses
Þnon_trainable_variables
ßlayers
 
 
 
µ
èregularization_losses
àmetrics
é	variables
êtrainable_variables
álayer_metrics
 âlayer_regularization_losses
ãnon_trainable_variables
älayers
 
 
 
 
*
N0
O1
P2
Q3
R4
S5
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

åtotal

æcount
ç	variables
è	keras_api
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

0
1

0
1
µ
¡regularization_losses
émetrics
¢	variables
£trainable_variables
êlayer_metrics
 ëlayer_regularization_losses
ìnon_trainable_variables
ílayers
 
 
 

0
1

0
1
µ
§regularization_losses
îmetrics
¨	variables
©trainable_variables
ïlayer_metrics
 ðlayer_regularization_losses
ñnon_trainable_variables
òlayers
 
 
 

0
1

0
1
µ
­regularization_losses
ómetrics
®	variables
¯trainable_variables
ôlayer_metrics
 õlayer_regularization_losses
önon_trainable_variables
÷layers
 
 
 
µ
±regularization_losses
ømetrics
²	variables
³trainable_variables
ùlayer_metrics
 úlayer_regularization_losses
ûnon_trainable_variables
ülayers
 
 
 
µ
µregularization_losses
ýmetrics
¶	variables
·trainable_variables
þlayer_metrics
 ÿlayer_regularization_losses
non_trainable_variables
layers
 
 
 

0
1

0
1
µ
»regularization_losses
metrics
¼	variables
½trainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
 
 
 
 
0
Ê0
Ë1
Ì2
Í3
Î4
Ï5
 

0
1

0
1
µ
Äregularization_losses
metrics
Å	variables
Ætrainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
 

0
1

0
1
µ
Èregularization_losses
metrics
É	variables
Êtrainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
 
 
 
 

Ô0
Õ1
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
å0
æ1

ç	variables
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

VARIABLE_VALUESGD/conv1d/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d_1/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/conv1d_1/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&SGD/batch_normalization/gamma/momentumXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE%SGD/batch_normalization/beta/momentumWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE(SGD/batch_normalization_1/gamma/momentumXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'SGD/batch_normalization_1/beta/momentumWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_4/kernel/momentumYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_4/bias/momentumWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_5/kernel/momentumYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_5/bias/momentumWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_6/kernel/momentumYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUESGD/dense_6/bias/momentumWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE>SGD/token_and_position_embedding/embedding/embeddings/momentumIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 
VARIABLE_VALUE@SGD/token_and_position_embedding/embedding_1/embeddings/momentumIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¥¢
VARIABLE_VALUEDSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentumJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEBSGD/transformer_block_1/multi_head_attention_1/query/bias/momentumJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEBSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentumJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¡
VARIABLE_VALUE@SGD/transformer_block_1/multi_head_attention_1/key/bias/momentumJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¥¢
VARIABLE_VALUEDSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentumJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
£ 
VARIABLE_VALUEBSGD/transformer_block_1/multi_head_attention_1/value/bias/momentumJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
°­
VARIABLE_VALUEOSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentumJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
®«
VARIABLE_VALUEMSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentumJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUESGD/dense_2/kernel/momentumJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUESGD/dense_2/bias/momentumJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUESGD/dense_3/kernel/momentumJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUESGD/dense_3/bias/momentumJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<SGD/transformer_block_1/layer_normalization_2/gamma/momentumJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE;SGD/transformer_block_1/layer_normalization_2/beta/momentumJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE<SGD/transformer_block_1/layer_normalization_3/gamma/momentumJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE;SGD/transformer_block_1/layer_normalization_3/beta/momentumJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿR
z
serving_default_input_2Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ã
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_23token_and_position_embedding/embedding_1/embeddings1token_and_position_embedding/embedding/embeddingsconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/beta%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/beta7transformer_block_1/multi_head_attention_1/query/kernel5transformer_block_1/multi_head_attention_1/query/bias5transformer_block_1/multi_head_attention_1/key/kernel3transformer_block_1/multi_head_attention_1/key/bias7transformer_block_1/multi_head_attention_1/value/kernel5transformer_block_1/multi_head_attention_1/value/biasBtransformer_block_1/multi_head_attention_1/attention_output/kernel@transformer_block_1/multi_head_attention_1/attention_output/bias/transformer_block_1/layer_normalization_2/gamma.transformer_block_1/layer_normalization_2/betadense_2/kerneldense_2/biasdense_3/kerneldense_3/bias/transformer_block_1/layer_normalization_3/gamma.transformer_block_1/layer_normalization_3/betadense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/bias*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_140701
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
$
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpEtoken_and_position_embedding/embedding/embeddings/Read/ReadVariableOpGtoken_and_position_embedding/embedding_1/embeddings/Read/ReadVariableOpKtransformer_block_1/multi_head_attention_1/query/kernel/Read/ReadVariableOpItransformer_block_1/multi_head_attention_1/query/bias/Read/ReadVariableOpItransformer_block_1/multi_head_attention_1/key/kernel/Read/ReadVariableOpGtransformer_block_1/multi_head_attention_1/key/bias/Read/ReadVariableOpKtransformer_block_1/multi_head_attention_1/value/kernel/Read/ReadVariableOpItransformer_block_1/multi_head_attention_1/value/bias/Read/ReadVariableOpVtransformer_block_1/multi_head_attention_1/attention_output/kernel/Read/ReadVariableOpTtransformer_block_1/multi_head_attention_1/attention_output/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpCtransformer_block_1/layer_normalization_2/gamma/Read/ReadVariableOpBtransformer_block_1/layer_normalization_2/beta/Read/ReadVariableOpCtransformer_block_1/layer_normalization_3/gamma/Read/ReadVariableOpBtransformer_block_1/layer_normalization_3/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.SGD/conv1d/kernel/momentum/Read/ReadVariableOp,SGD/conv1d/bias/momentum/Read/ReadVariableOp0SGD/conv1d_1/kernel/momentum/Read/ReadVariableOp.SGD/conv1d_1/bias/momentum/Read/ReadVariableOp:SGD/batch_normalization/gamma/momentum/Read/ReadVariableOp9SGD/batch_normalization/beta/momentum/Read/ReadVariableOp<SGD/batch_normalization_1/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_1/beta/momentum/Read/ReadVariableOp/SGD/dense_4/kernel/momentum/Read/ReadVariableOp-SGD/dense_4/bias/momentum/Read/ReadVariableOp/SGD/dense_5/kernel/momentum/Read/ReadVariableOp-SGD/dense_5/bias/momentum/Read/ReadVariableOp/SGD/dense_6/kernel/momentum/Read/ReadVariableOp-SGD/dense_6/bias/momentum/Read/ReadVariableOpRSGD/token_and_position_embedding/embedding/embeddings/momentum/Read/ReadVariableOpTSGD/token_and_position_embedding/embedding_1/embeddings/momentum/Read/ReadVariableOpXSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_1/multi_head_attention_1/query/bias/momentum/Read/ReadVariableOpVSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentum/Read/ReadVariableOpTSGD/transformer_block_1/multi_head_attention_1/key/bias/momentum/Read/ReadVariableOpXSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_1/multi_head_attention_1/value/bias/momentum/Read/ReadVariableOpcSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentum/Read/ReadVariableOpaSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentum/Read/ReadVariableOp/SGD/dense_2/kernel/momentum/Read/ReadVariableOp-SGD/dense_2/bias/momentum/Read/ReadVariableOp/SGD/dense_3/kernel/momentum/Read/ReadVariableOp-SGD/dense_3/bias/momentum/Read/ReadVariableOpPSGD/transformer_block_1/layer_normalization_2/gamma/momentum/Read/ReadVariableOpOSGD/transformer_block_1/layer_normalization_2/beta/momentum/Read/ReadVariableOpPSGD/transformer_block_1/layer_normalization_3/gamma/momentum/Read/ReadVariableOpOSGD/transformer_block_1/layer_normalization_3/beta/momentum/Read/ReadVariableOpConst*W
TinP
N2L	*
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_142784
Ã
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdecaylearning_ratemomentumSGD/iter1token_and_position_embedding/embedding/embeddings3token_and_position_embedding/embedding_1/embeddings7transformer_block_1/multi_head_attention_1/query/kernel5transformer_block_1/multi_head_attention_1/query/bias5transformer_block_1/multi_head_attention_1/key/kernel3transformer_block_1/multi_head_attention_1/key/bias7transformer_block_1/multi_head_attention_1/value/kernel5transformer_block_1/multi_head_attention_1/value/biasBtransformer_block_1/multi_head_attention_1/attention_output/kernel@transformer_block_1/multi_head_attention_1/attention_output/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias/transformer_block_1/layer_normalization_2/gamma.transformer_block_1/layer_normalization_2/beta/transformer_block_1/layer_normalization_3/gamma.transformer_block_1/layer_normalization_3/betatotalcountSGD/conv1d/kernel/momentumSGD/conv1d/bias/momentumSGD/conv1d_1/kernel/momentumSGD/conv1d_1/bias/momentum&SGD/batch_normalization/gamma/momentum%SGD/batch_normalization/beta/momentum(SGD/batch_normalization_1/gamma/momentum'SGD/batch_normalization_1/beta/momentumSGD/dense_4/kernel/momentumSGD/dense_4/bias/momentumSGD/dense_5/kernel/momentumSGD/dense_5/bias/momentumSGD/dense_6/kernel/momentumSGD/dense_6/bias/momentum>SGD/token_and_position_embedding/embedding/embeddings/momentum@SGD/token_and_position_embedding/embedding_1/embeddings/momentumDSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentumBSGD/transformer_block_1/multi_head_attention_1/query/bias/momentumBSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentum@SGD/transformer_block_1/multi_head_attention_1/key/bias/momentumDSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentumBSGD/transformer_block_1/multi_head_attention_1/value/bias/momentumOSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentumMSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentumSGD/dense_2/kernel/momentumSGD/dense_2/bias/momentumSGD/dense_3/kernel/momentumSGD/dense_3/bias/momentum<SGD/transformer_block_1/layer_normalization_2/gamma/momentum;SGD/transformer_block_1/layer_normalization_2/beta/momentum<SGD/transformer_block_1/layer_normalization_3/gamma/momentum;SGD/transformer_block_1/layer_normalization_3/beta/momentum*V
TinO
M2K*
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_143016îü%
¸
 
-__inference_sequential_1_layer_call_fn_142446

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_1392742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

÷
D__inference_conv1d_1_layer_call_and_return_conditional_losses_139406

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÞ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 
 
_user_specified_nameinputs
° 
â
C__inference_dense_2_layer_call_and_return_conditional_losses_139180

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: @*
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ# ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ý
'
!__inference__wrapped_model_138820
input_1
input_2J
Fmodel_token_and_position_embedding_embedding_1_embedding_lookup_138589H
Dmodel_token_and_position_embedding_embedding_embedding_lookup_138595<
8model_conv1d_conv1d_expanddims_1_readvariableop_resource0
,model_conv1d_biasadd_readvariableop_resource>
:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource2
.model_conv1d_1_biasadd_readvariableop_resource?
;model_batch_normalization_batchnorm_readvariableop_resourceC
?model_batch_normalization_batchnorm_mul_readvariableop_resourceA
=model_batch_normalization_batchnorm_readvariableop_1_resourceA
=model_batch_normalization_batchnorm_readvariableop_2_resourceA
=model_batch_normalization_1_batchnorm_readvariableop_resourceE
Amodel_batch_normalization_1_batchnorm_mul_readvariableop_resourceC
?model_batch_normalization_1_batchnorm_readvariableop_1_resourceC
?model_batch_normalization_1_batchnorm_readvariableop_2_resource`
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
identity¢2model/batch_normalization/batchnorm/ReadVariableOp¢4model/batch_normalization/batchnorm/ReadVariableOp_1¢4model/batch_normalization/batchnorm/ReadVariableOp_2¢6model/batch_normalization/batchnorm/mul/ReadVariableOp¢4model/batch_normalization_1/batchnorm/ReadVariableOp¢6model/batch_normalization_1/batchnorm/ReadVariableOp_1¢6model/batch_normalization_1/batchnorm/ReadVariableOp_2¢8model/batch_normalization_1/batchnorm/mul/ReadVariableOp¢#model/conv1d/BiasAdd/ReadVariableOp¢/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp¢%model/conv1d_1/BiasAdd/ReadVariableOp¢1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp¢$model/dense_4/BiasAdd/ReadVariableOp¢#model/dense_4/MatMul/ReadVariableOp¢$model/dense_5/BiasAdd/ReadVariableOp¢#model/dense_5/MatMul/ReadVariableOp¢$model/dense_6/BiasAdd/ReadVariableOp¢#model/dense_6/MatMul/ReadVariableOp¢=model/token_and_position_embedding/embedding/embedding_lookup¢?model/token_and_position_embedding/embedding_1/embedding_lookup¢Hmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp¢Lmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp¢Hmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp¢Lmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp¢Tmodel/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp¢^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp¢Gmodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOp¢Qmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp¢Imodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOp¢Smodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp¢Imodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOp¢Smodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp¢Emodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp¢Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp¢Emodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp¢Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp
(model/token_and_position_embedding/ShapeShapeinput_1*
T0*
_output_shapes
:2*
(model/token_and_position_embedding/ShapeÃ
6model/token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ28
6model/token_and_position_embedding/strided_slice/stack¾
8model/token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8model/token_and_position_embedding/strided_slice/stack_1¾
8model/token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/token_and_position_embedding/strided_slice/stack_2´
0model/token_and_position_embedding/strided_sliceStridedSlice1model/token_and_position_embedding/Shape:output:0?model/token_and_position_embedding/strided_slice/stack:output:0Amodel/token_and_position_embedding/strided_slice/stack_1:output:0Amodel/token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0model/token_and_position_embedding/strided_slice¢
.model/token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : 20
.model/token_and_position_embedding/range/start¢
.model/token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :20
.model/token_and_position_embedding/range/delta¯
(model/token_and_position_embedding/rangeRange7model/token_and_position_embedding/range/start:output:09model/token_and_position_embedding/strided_slice:output:07model/token_and_position_embedding/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(model/token_and_position_embedding/rangeÞ
?model/token_and_position_embedding/embedding_1/embedding_lookupResourceGatherFmodel_token_and_position_embedding_embedding_1_embedding_lookup_1385891model/token_and_position_embedding/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Y
_classO
MKloc:@model/token_and_position_embedding/embedding_1/embedding_lookup/138589*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02A
?model/token_and_position_embedding/embedding_1/embedding_lookup¥
Hmodel/token_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityHmodel/token_and_position_embedding/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Y
_classO
MKloc:@model/token_and_position_embedding/embedding_1/embedding_lookup/138589*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2J
Hmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity©
Jmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityQmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2L
Jmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity_1¹
1model/token_and_position_embedding/embedding/CastCastinput_1*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR23
1model/token_and_position_embedding/embedding/Castß
=model/token_and_position_embedding/embedding/embedding_lookupResourceGatherDmodel_token_and_position_embedding_embedding_embedding_lookup_1385955model/token_and_position_embedding/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*W
_classM
KIloc:@model/token_and_position_embedding/embedding/embedding_lookup/138595*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02?
=model/token_and_position_embedding/embedding/embedding_lookup¢
Fmodel/token_and_position_embedding/embedding/embedding_lookup/IdentityIdentityFmodel/token_and_position_embedding/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@model/token_and_position_embedding/embedding/embedding_lookup/138595*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2H
Fmodel/token_and_position_embedding/embedding/embedding_lookup/Identity¨
Hmodel/token_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityOmodel/token_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2J
Hmodel/token_and_position_embedding/embedding/embedding_lookup/Identity_1¸
&model/token_and_position_embedding/addAddV2Qmodel/token_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Smodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2(
&model/token_and_position_embedding/add
"model/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2$
"model/conv1d/conv1d/ExpandDims/dimâ
model/conv1d/conv1d/ExpandDims
ExpandDims*model/token_and_position_embedding/add:z:0+model/conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2 
model/conv1d/conv1d/ExpandDimsß
/model/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8model_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype021
/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp
$model/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$model/conv1d/conv1d/ExpandDims_1/dimë
 model/conv1d/conv1d/ExpandDims_1
ExpandDims7model/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0-model/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2"
 model/conv1d/conv1d/ExpandDims_1ë
model/conv1d/conv1dConv2D'model/conv1d/conv1d/ExpandDims:output:0)model/conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
paddingSAME*
strides
2
model/conv1d/conv1dº
model/conv1d/conv1d/SqueezeSqueezemodel/conv1d/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
model/conv1d/conv1d/Squeeze³
#model/conv1d/BiasAdd/ReadVariableOpReadVariableOp,model_conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#model/conv1d/BiasAdd/ReadVariableOpÁ
model/conv1d/BiasAddBiasAdd$model/conv1d/conv1d/Squeeze:output:0+model/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
model/conv1d/BiasAdd
model/conv1d/ReluRelumodel/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
model/conv1d/Relu
&model/average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&model/average_pooling1d/ExpandDims/dimã
"model/average_pooling1d/ExpandDims
ExpandDimsmodel/conv1d/Relu:activations:0/model/average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2$
"model/average_pooling1d/ExpandDimsñ
model/average_pooling1d/AvgPoolAvgPool+model/average_pooling1d/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
ksize
*
paddingVALID*
strides
2!
model/average_pooling1d/AvgPoolÅ
model/average_pooling1d/SqueezeSqueeze(model/average_pooling1d/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims
2!
model/average_pooling1d/Squeeze
$model/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2&
$model/conv1d_1/conv1d/ExpandDims/dimæ
 model/conv1d_1/conv1d/ExpandDims
ExpandDims(model/average_pooling1d/Squeeze:output:0-model/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2"
 model/conv1d_1/conv1d/ExpandDimså
1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype023
1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp
&model/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/conv1d_1/conv1d/ExpandDims_1/dimó
"model/conv1d_1/conv1d/ExpandDims_1
ExpandDims9model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2$
"model/conv1d_1/conv1d/ExpandDims_1ó
model/conv1d_1/conv1dConv2D)model/conv1d_1/conv1d/ExpandDims:output:0+model/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
model/conv1d_1/conv1dÀ
model/conv1d_1/conv1d/SqueezeSqueezemodel/conv1d_1/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
model/conv1d_1/conv1d/Squeeze¹
%model/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model/conv1d_1/BiasAdd/ReadVariableOpÉ
model/conv1d_1/BiasAddBiasAdd&model/conv1d_1/conv1d/Squeeze:output:0-model/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
model/conv1d_1/BiasAdd
model/conv1d_1/ReluRelumodel/conv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
model/conv1d_1/Relu
(model/average_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(model/average_pooling1d_2/ExpandDims/dimô
$model/average_pooling1d_2/ExpandDims
ExpandDims*model/token_and_position_embedding/add:z:01model/average_pooling1d_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2&
$model/average_pooling1d_2/ExpandDimsø
!model/average_pooling1d_2/AvgPoolAvgPool-model/average_pooling1d_2/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize	
¬*
paddingVALID*
strides	
¬2#
!model/average_pooling1d_2/AvgPoolÊ
!model/average_pooling1d_2/SqueezeSqueeze*model/average_pooling1d_2/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2#
!model/average_pooling1d_2/Squeeze
(model/average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(model/average_pooling1d_1/ExpandDims/dimë
$model/average_pooling1d_1/ExpandDims
ExpandDims!model/conv1d_1/Relu:activations:01model/average_pooling1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2&
$model/average_pooling1d_1/ExpandDimsö
!model/average_pooling1d_1/AvgPoolAvgPool-model/average_pooling1d_1/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize

*
paddingVALID*
strides

2#
!model/average_pooling1d_1/AvgPoolÊ
!model/average_pooling1d_1/SqueezeSqueeze*model/average_pooling1d_1/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2#
!model/average_pooling1d_1/Squeezeà
2model/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype024
2model/batch_normalization/batchnorm/ReadVariableOp
)model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2+
)model/batch_normalization/batchnorm/add/yð
'model/batch_normalization/batchnorm/addAddV2:model/batch_normalization/batchnorm/ReadVariableOp:value:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2)
'model/batch_normalization/batchnorm/add±
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2+
)model/batch_normalization/batchnorm/Rsqrtì
6model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype028
6model/batch_normalization/batchnorm/mul/ReadVariableOpí
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:0>model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2)
'model/batch_normalization/batchnorm/mulì
)model/batch_normalization/batchnorm/mul_1Mul*model/average_pooling1d_1/Squeeze:output:0+model/batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2+
)model/batch_normalization/batchnorm/mul_1æ
4model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype026
4model/batch_normalization/batchnorm/ReadVariableOp_1í
)model/batch_normalization/batchnorm/mul_2Mul<model/batch_normalization/batchnorm/ReadVariableOp_1:value:0+model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2+
)model/batch_normalization/batchnorm/mul_2æ
4model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype026
4model/batch_normalization/batchnorm/ReadVariableOp_2ë
'model/batch_normalization/batchnorm/subSub<model/batch_normalization/batchnorm/ReadVariableOp_2:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2)
'model/batch_normalization/batchnorm/subñ
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2+
)model/batch_normalization/batchnorm/add_1æ
4model/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype026
4model/batch_normalization_1/batchnorm/ReadVariableOp
+model/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2-
+model/batch_normalization_1/batchnorm/add/yø
)model/batch_normalization_1/batchnorm/addAddV2<model/batch_normalization_1/batchnorm/ReadVariableOp:value:04model/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2+
)model/batch_normalization_1/batchnorm/add·
+model/batch_normalization_1/batchnorm/RsqrtRsqrt-model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
: 2-
+model/batch_normalization_1/batchnorm/Rsqrtò
8model/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02:
8model/batch_normalization_1/batchnorm/mul/ReadVariableOpõ
)model/batch_normalization_1/batchnorm/mulMul/model/batch_normalization_1/batchnorm/Rsqrt:y:0@model/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2+
)model/batch_normalization_1/batchnorm/mulò
+model/batch_normalization_1/batchnorm/mul_1Mul*model/average_pooling1d_2/Squeeze:output:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+model/batch_normalization_1/batchnorm/mul_1ì
6model/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype028
6model/batch_normalization_1/batchnorm/ReadVariableOp_1õ
+model/batch_normalization_1/batchnorm/mul_2Mul>model/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
: 2-
+model/batch_normalization_1/batchnorm/mul_2ì
6model/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype028
6model/batch_normalization_1/batchnorm/ReadVariableOp_2ó
)model/batch_normalization_1/batchnorm/subSub>model/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2+
)model/batch_normalization_1/batchnorm/subù
+model/batch_normalization_1/batchnorm/add_1AddV2/model/batch_normalization_1/batchnorm/mul_1:z:0-model/batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+model/batch_normalization_1/batchnorm/add_1½
model/add/addAddV2-model/batch_normalization/batchnorm/add_1:z:0/model/batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
model/add/addË
Smodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOp\model_transformer_block_1_multi_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02U
Smodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpæ
Dmodel/transformer_block_1/multi_head_attention_1/query/einsum/EinsumEinsummodel/add/add:z:0[model/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2F
Dmodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum©
Imodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpReadVariableOpRmodel_transformer_block_1_multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype02K
Imodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpÝ
:model/transformer_block_1/multi_head_attention_1/query/addAddV2Mmodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum:output:0Qmodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:model/transformer_block_1/multi_head_attention_1/query/addÅ
Qmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOpZmodel_transformer_block_1_multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02S
Qmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpà
Bmodel/transformer_block_1/multi_head_attention_1/key/einsum/EinsumEinsummodel/add/add:z:0Ymodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2D
Bmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum£
Gmodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOpReadVariableOpPmodel_transformer_block_1_multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02I
Gmodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOpÕ
8model/transformer_block_1/multi_head_attention_1/key/addAddV2Kmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum:output:0Omodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8model/transformer_block_1/multi_head_attention_1/key/addË
Smodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOp\model_transformer_block_1_multi_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02U
Smodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpæ
Dmodel/transformer_block_1/multi_head_attention_1/value/einsum/EinsumEinsummodel/add/add:z:0[model/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2F
Dmodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum©
Imodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpReadVariableOpRmodel_transformer_block_1_multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype02K
Imodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpÝ
:model/transformer_block_1/multi_head_attention_1/value/addAddV2Mmodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum:output:0Qmodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2<
:model/transformer_block_1/multi_head_attention_1/value/addµ
6model/transformer_block_1/multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>28
6model/transformer_block_1/multi_head_attention_1/Mul/y®
4model/transformer_block_1/multi_head_attention_1/MulMul>model/transformer_block_1/multi_head_attention_1/query/add:z:0?model/transformer_block_1/multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4model/transformer_block_1/multi_head_attention_1/Mulä
>model/transformer_block_1/multi_head_attention_1/einsum/EinsumEinsum<model/transformer_block_1/multi_head_attention_1/key/add:z:08model/transformer_block_1/multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2@
>model/transformer_block_1/multi_head_attention_1/einsum/Einsum
@model/transformer_block_1/multi_head_attention_1/softmax/SoftmaxSoftmaxGmodel/transformer_block_1/multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2B
@model/transformer_block_1/multi_head_attention_1/softmax/Softmax
Amodel/transformer_block_1/multi_head_attention_1/dropout/IdentityIdentityJmodel/transformer_block_1/multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2C
Amodel/transformer_block_1/multi_head_attention_1/dropout/Identityü
@model/transformer_block_1/multi_head_attention_1/einsum_1/EinsumEinsumJmodel/transformer_block_1/multi_head_attention_1/dropout/Identity:output:0>model/transformer_block_1/multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2B
@model/transformer_block_1/multi_head_attention_1/einsum_1/Einsumì
^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_transformer_block_1_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02`
^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp»
Omodel/transformer_block_1/multi_head_attention_1/attention_output/einsum/EinsumEinsumImodel/transformer_block_1/multi_head_attention_1/einsum_1/Einsum:output:0fmodel/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe2Q
Omodel/transformer_block_1/multi_head_attention_1/attention_output/einsum/EinsumÆ
Tmodel/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOp]model_transformer_block_1_multi_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02V
Tmodel/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp
Emodel/transformer_block_1/multi_head_attention_1/attention_output/addAddV2Xmodel/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum:output:0\model/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2G
Emodel/transformer_block_1/multi_head_attention_1/attention_output/addé
,model/transformer_block_1/dropout_2/IdentityIdentityImodel/transformer_block_1/multi_head_attention_1/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2.
,model/transformer_block_1/dropout_2/IdentityÇ
model/transformer_block_1/addAddV2model/add/add:z:05model/transformer_block_1/dropout_2/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
model/transformer_block_1/addê
Nmodel/transformer_block_1/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
Nmodel/transformer_block_1/layer_normalization_2/moments/mean/reduction_indicesÇ
<model/transformer_block_1/layer_normalization_2/moments/meanMean!model/transformer_block_1/add:z:0Wmodel/transformer_block_1/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2>
<model/transformer_block_1/layer_normalization_2/moments/mean
Dmodel/transformer_block_1/layer_normalization_2/moments/StopGradientStopGradientEmodel/transformer_block_1/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2F
Dmodel/transformer_block_1/layer_normalization_2/moments/StopGradientÓ
Imodel/transformer_block_1/layer_normalization_2/moments/SquaredDifferenceSquaredDifference!model/transformer_block_1/add:z:0Mmodel/transformer_block_1/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2K
Imodel/transformer_block_1/layer_normalization_2/moments/SquaredDifferenceò
Rmodel/transformer_block_1/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2T
Rmodel/transformer_block_1/layer_normalization_2/moments/variance/reduction_indicesÿ
@model/transformer_block_1/layer_normalization_2/moments/varianceMeanMmodel/transformer_block_1/layer_normalization_2/moments/SquaredDifference:z:0[model/transformer_block_1/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2B
@model/transformer_block_1/layer_normalization_2/moments/varianceÇ
?model/transformer_block_1/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752A
?model/transformer_block_1/layer_normalization_2/batchnorm/add/yÒ
=model/transformer_block_1/layer_normalization_2/batchnorm/addAddV2Imodel/transformer_block_1/layer_normalization_2/moments/variance:output:0Hmodel/transformer_block_1/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2?
=model/transformer_block_1/layer_normalization_2/batchnorm/add
?model/transformer_block_1/layer_normalization_2/batchnorm/RsqrtRsqrtAmodel/transformer_block_1/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2A
?model/transformer_block_1/layer_normalization_2/batchnorm/Rsqrt®
Lmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpUmodel_transformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02N
Lmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpÖ
=model/transformer_block_1/layer_normalization_2/batchnorm/mulMulCmodel/transformer_block_1/layer_normalization_2/batchnorm/Rsqrt:y:0Tmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2?
=model/transformer_block_1/layer_normalization_2/batchnorm/mul¥
?model/transformer_block_1/layer_normalization_2/batchnorm/mul_1Mul!model/transformer_block_1/add:z:0Amodel/transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?model/transformer_block_1/layer_normalization_2/batchnorm/mul_1É
?model/transformer_block_1/layer_normalization_2/batchnorm/mul_2MulEmodel/transformer_block_1/layer_normalization_2/moments/mean:output:0Amodel/transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?model/transformer_block_1/layer_normalization_2/batchnorm/mul_2¢
Hmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpQmodel_transformer_block_1_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpÒ
=model/transformer_block_1/layer_normalization_2/batchnorm/subSubPmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp:value:0Cmodel/transformer_block_1/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2?
=model/transformer_block_1/layer_normalization_2/batchnorm/subÉ
?model/transformer_block_1/layer_normalization_2/batchnorm/add_1AddV2Cmodel/transformer_block_1/layer_normalization_2/batchnorm/mul_1:z:0Amodel/transformer_block_1/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?model/transformer_block_1/layer_normalization_2/batchnorm/add_1£
Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOpPmodel_transformer_block_1_sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02I
Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpÈ
=model/transformer_block_1/sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2?
=model/transformer_block_1/sequential_1/dense_2/Tensordot/axesÏ
=model/transformer_block_1/sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model/transformer_block_1/sequential_1/dense_2/Tensordot/freeó
>model/transformer_block_1/sequential_1/dense_2/Tensordot/ShapeShapeCmodel/transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2@
>model/transformer_block_1/sequential_1/dense_2/Tensordot/ShapeÒ
Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axis¼
Amodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2GatherV2Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0Omodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Amodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2Ö
Hmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axisÂ
Cmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1GatherV2Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Qmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1Ê
>model/transformer_block_1/sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2@
>model/transformer_block_1/sequential_1/dense_2/Tensordot/Const¼
=model/transformer_block_1/sequential_1/dense_2/Tensordot/ProdProdJmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2?
=model/transformer_block_1/sequential_1/dense_2/Tensordot/ProdÎ
@model/transformer_block_1/sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block_1/sequential_1/dense_2/Tensordot/Const_1Ä
?model/transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1ProdLmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1:output:0Imodel/transformer_block_1/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2A
?model/transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1Î
Dmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat/axis
?model/transformer_block_1/sequential_1/dense_2/Tensordot/concatConcatV2Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Mmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2A
?model/transformer_block_1/sequential_1/dense_2/Tensordot/concatÈ
>model/transformer_block_1/sequential_1/dense_2/Tensordot/stackPackFmodel/transformer_block_1/sequential_1/dense_2/Tensordot/Prod:output:0Hmodel/transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2@
>model/transformer_block_1/sequential_1/dense_2/Tensordot/stackÚ
Bmodel/transformer_block_1/sequential_1/dense_2/Tensordot/transpose	TransposeCmodel/transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0Hmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2D
Bmodel/transformer_block_1/sequential_1/dense_2/Tensordot/transposeÛ
@model/transformer_block_1/sequential_1/dense_2/Tensordot/ReshapeReshapeFmodel/transformer_block_1/sequential_1/dense_2/Tensordot/transpose:y:0Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2B
@model/transformer_block_1/sequential_1/dense_2/Tensordot/ReshapeÚ
?model/transformer_block_1/sequential_1/dense_2/Tensordot/MatMulMatMulImodel/transformer_block_1/sequential_1/dense_2/Tensordot/Reshape:output:0Omodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2A
?model/transformer_block_1/sequential_1/dense_2/Tensordot/MatMulÎ
@model/transformer_block_1/sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2B
@model/transformer_block_1/sequential_1/dense_2/Tensordot/Const_2Ò
Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axis¨
Amodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1ConcatV2Jmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Imodel/transformer_block_1/sequential_1/dense_2/Tensordot/Const_2:output:0Omodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1Ì
8model/transformer_block_1/sequential_1/dense_2/TensordotReshapeImodel/transformer_block_1/sequential_1/dense_2/Tensordot/MatMul:product:0Jmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2:
8model/transformer_block_1/sequential_1/dense_2/Tensordot
Emodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOpNmodel_transformer_block_1_sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02G
Emodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpÃ
6model/transformer_block_1/sequential_1/dense_2/BiasAddBiasAddAmodel/transformer_block_1/sequential_1/dense_2/Tensordot:output:0Mmodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@28
6model/transformer_block_1/sequential_1/dense_2/BiasAddé
3model/transformer_block_1/sequential_1/dense_2/ReluRelu?model/transformer_block_1/sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@25
3model/transformer_block_1/sequential_1/dense_2/Relu£
Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOpPmodel_transformer_block_1_sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02I
Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpÈ
=model/transformer_block_1/sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2?
=model/transformer_block_1/sequential_1/dense_3/Tensordot/axesÏ
=model/transformer_block_1/sequential_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model/transformer_block_1/sequential_1/dense_3/Tensordot/freeñ
>model/transformer_block_1/sequential_1/dense_3/Tensordot/ShapeShapeAmodel/transformer_block_1/sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2@
>model/transformer_block_1/sequential_1/dense_3/Tensordot/ShapeÒ
Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axis¼
Amodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2GatherV2Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0Omodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Amodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2Ö
Hmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axisÂ
Cmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1GatherV2Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Qmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1Ê
>model/transformer_block_1/sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2@
>model/transformer_block_1/sequential_1/dense_3/Tensordot/Const¼
=model/transformer_block_1/sequential_1/dense_3/Tensordot/ProdProdJmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2?
=model/transformer_block_1/sequential_1/dense_3/Tensordot/ProdÎ
@model/transformer_block_1/sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block_1/sequential_1/dense_3/Tensordot/Const_1Ä
?model/transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1ProdLmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1:output:0Imodel/transformer_block_1/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2A
?model/transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1Î
Dmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat/axis
?model/transformer_block_1/sequential_1/dense_3/Tensordot/concatConcatV2Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Mmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2A
?model/transformer_block_1/sequential_1/dense_3/Tensordot/concatÈ
>model/transformer_block_1/sequential_1/dense_3/Tensordot/stackPackFmodel/transformer_block_1/sequential_1/dense_3/Tensordot/Prod:output:0Hmodel/transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2@
>model/transformer_block_1/sequential_1/dense_3/Tensordot/stackØ
Bmodel/transformer_block_1/sequential_1/dense_3/Tensordot/transpose	TransposeAmodel/transformer_block_1/sequential_1/dense_2/Relu:activations:0Hmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2D
Bmodel/transformer_block_1/sequential_1/dense_3/Tensordot/transposeÛ
@model/transformer_block_1/sequential_1/dense_3/Tensordot/ReshapeReshapeFmodel/transformer_block_1/sequential_1/dense_3/Tensordot/transpose:y:0Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2B
@model/transformer_block_1/sequential_1/dense_3/Tensordot/ReshapeÚ
?model/transformer_block_1/sequential_1/dense_3/Tensordot/MatMulMatMulImodel/transformer_block_1/sequential_1/dense_3/Tensordot/Reshape:output:0Omodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2A
?model/transformer_block_1/sequential_1/dense_3/Tensordot/MatMulÎ
@model/transformer_block_1/sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block_1/sequential_1/dense_3/Tensordot/Const_2Ò
Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axis¨
Amodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1ConcatV2Jmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Imodel/transformer_block_1/sequential_1/dense_3/Tensordot/Const_2:output:0Omodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1Ì
8model/transformer_block_1/sequential_1/dense_3/TensordotReshapeImodel/transformer_block_1/sequential_1/dense_3/Tensordot/MatMul:product:0Jmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2:
8model/transformer_block_1/sequential_1/dense_3/Tensordot
Emodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOpNmodel_transformer_block_1_sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02G
Emodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpÃ
6model/transformer_block_1/sequential_1/dense_3/BiasAddBiasAddAmodel/transformer_block_1/sequential_1/dense_3/Tensordot:output:0Mmodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 28
6model/transformer_block_1/sequential_1/dense_3/BiasAddß
,model/transformer_block_1/dropout_3/IdentityIdentity?model/transformer_block_1/sequential_1/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2.
,model/transformer_block_1/dropout_3/Identityý
model/transformer_block_1/add_1AddV2Cmodel/transformer_block_1/layer_normalization_2/batchnorm/add_1:z:05model/transformer_block_1/dropout_3/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2!
model/transformer_block_1/add_1ê
Nmodel/transformer_block_1/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
Nmodel/transformer_block_1/layer_normalization_3/moments/mean/reduction_indicesÉ
<model/transformer_block_1/layer_normalization_3/moments/meanMean#model/transformer_block_1/add_1:z:0Wmodel/transformer_block_1/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2>
<model/transformer_block_1/layer_normalization_3/moments/mean
Dmodel/transformer_block_1/layer_normalization_3/moments/StopGradientStopGradientEmodel/transformer_block_1/layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2F
Dmodel/transformer_block_1/layer_normalization_3/moments/StopGradientÕ
Imodel/transformer_block_1/layer_normalization_3/moments/SquaredDifferenceSquaredDifference#model/transformer_block_1/add_1:z:0Mmodel/transformer_block_1/layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2K
Imodel/transformer_block_1/layer_normalization_3/moments/SquaredDifferenceò
Rmodel/transformer_block_1/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2T
Rmodel/transformer_block_1/layer_normalization_3/moments/variance/reduction_indicesÿ
@model/transformer_block_1/layer_normalization_3/moments/varianceMeanMmodel/transformer_block_1/layer_normalization_3/moments/SquaredDifference:z:0[model/transformer_block_1/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2B
@model/transformer_block_1/layer_normalization_3/moments/varianceÇ
?model/transformer_block_1/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752A
?model/transformer_block_1/layer_normalization_3/batchnorm/add/yÒ
=model/transformer_block_1/layer_normalization_3/batchnorm/addAddV2Imodel/transformer_block_1/layer_normalization_3/moments/variance:output:0Hmodel/transformer_block_1/layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2?
=model/transformer_block_1/layer_normalization_3/batchnorm/add
?model/transformer_block_1/layer_normalization_3/batchnorm/RsqrtRsqrtAmodel/transformer_block_1/layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2A
?model/transformer_block_1/layer_normalization_3/batchnorm/Rsqrt®
Lmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpUmodel_transformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02N
Lmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpÖ
=model/transformer_block_1/layer_normalization_3/batchnorm/mulMulCmodel/transformer_block_1/layer_normalization_3/batchnorm/Rsqrt:y:0Tmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2?
=model/transformer_block_1/layer_normalization_3/batchnorm/mul§
?model/transformer_block_1/layer_normalization_3/batchnorm/mul_1Mul#model/transformer_block_1/add_1:z:0Amodel/transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?model/transformer_block_1/layer_normalization_3/batchnorm/mul_1É
?model/transformer_block_1/layer_normalization_3/batchnorm/mul_2MulEmodel/transformer_block_1/layer_normalization_3/moments/mean:output:0Amodel/transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?model/transformer_block_1/layer_normalization_3/batchnorm/mul_2¢
Hmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOpQmodel_transformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpÒ
=model/transformer_block_1/layer_normalization_3/batchnorm/subSubPmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp:value:0Cmodel/transformer_block_1/layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2?
=model/transformer_block_1/layer_normalization_3/batchnorm/subÉ
?model/transformer_block_1/layer_normalization_3/batchnorm/add_1AddV2Cmodel/transformer_block_1/layer_normalization_3/batchnorm/mul_1:z:0Amodel/transformer_block_1/layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?model/transformer_block_1/layer_normalization_3/batchnorm/add_1{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
model/flatten/ConstÏ
model/flatten/ReshapeReshapeCmodel/transformer_block_1/layer_normalization_3/batchnorm/add_1:z:0model/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
model/flatten/Reshape
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axisÍ
model/concatenate/concatConcatV2model/flatten/Reshape:output:0input_2&model/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
model/concatenate/concat¸
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02%
#model/dense_4/MatMul/ReadVariableOp¸
model/dense_4/MatMulMatMul!model/concatenate/concat:output:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/dense_4/MatMul¶
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model/dense_4/BiasAdd/ReadVariableOp¹
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/dense_4/BiasAdd
model/dense_4/ReluRelumodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/dense_4/Relu
model/dropout_4/IdentityIdentity model/dense_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/dropout_4/Identity·
#model/dense_5/MatMul/ReadVariableOpReadVariableOp,model_dense_5_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02%
#model/dense_5/MatMul/ReadVariableOp¸
model/dense_5/MatMulMatMul!model/dropout_4/Identity:output:0+model/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/dense_5/MatMul¶
$model/dense_5/BiasAdd/ReadVariableOpReadVariableOp-model_dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model/dense_5/BiasAdd/ReadVariableOp¹
model/dense_5/BiasAddBiasAddmodel/dense_5/MatMul:product:0,model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/dense_5/BiasAdd
model/dense_5/ReluRelumodel/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/dense_5/Relu
model/dropout_5/IdentityIdentity model/dense_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/dropout_5/Identity·
#model/dense_6/MatMul/ReadVariableOpReadVariableOp,model_dense_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02%
#model/dense_6/MatMul/ReadVariableOp¸
model/dense_6/MatMulMatMul!model/dropout_5/Identity:output:0+model/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_6/MatMul¶
$model/dense_6/BiasAdd/ReadVariableOpReadVariableOp-model_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_6/BiasAdd/ReadVariableOp¹
model/dense_6/BiasAddBiasAddmodel/dense_6/MatMul:product:0,model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_6/BiasAddÁ
IdentityIdentitymodel/dense_6/BiasAdd:output:03^model/batch_normalization/batchnorm/ReadVariableOp5^model/batch_normalization/batchnorm/ReadVariableOp_15^model/batch_normalization/batchnorm/ReadVariableOp_27^model/batch_normalization/batchnorm/mul/ReadVariableOp5^model/batch_normalization_1/batchnorm/ReadVariableOp7^model/batch_normalization_1/batchnorm/ReadVariableOp_17^model/batch_normalization_1/batchnorm/ReadVariableOp_29^model/batch_normalization_1/batchnorm/mul/ReadVariableOp$^model/conv1d/BiasAdd/ReadVariableOp0^model/conv1d/conv1d/ExpandDims_1/ReadVariableOp&^model/conv1d_1/BiasAdd/ReadVariableOp2^model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp%^model/dense_5/BiasAdd/ReadVariableOp$^model/dense_5/MatMul/ReadVariableOp%^model/dense_6/BiasAdd/ReadVariableOp$^model/dense_6/MatMul/ReadVariableOp>^model/token_and_position_embedding/embedding/embedding_lookup@^model/token_and_position_embedding/embedding_1/embedding_lookupI^model/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpM^model/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpI^model/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpM^model/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpU^model/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp_^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpH^model/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOpR^model/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpJ^model/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpT^model/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpJ^model/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpT^model/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpF^model/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpH^model/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpF^model/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpH^model/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2h
2model/batch_normalization/batchnorm/ReadVariableOp2model/batch_normalization/batchnorm/ReadVariableOp2l
4model/batch_normalization/batchnorm/ReadVariableOp_14model/batch_normalization/batchnorm/ReadVariableOp_12l
4model/batch_normalization/batchnorm/ReadVariableOp_24model/batch_normalization/batchnorm/ReadVariableOp_22p
6model/batch_normalization/batchnorm/mul/ReadVariableOp6model/batch_normalization/batchnorm/mul/ReadVariableOp2l
4model/batch_normalization_1/batchnorm/ReadVariableOp4model/batch_normalization_1/batchnorm/ReadVariableOp2p
6model/batch_normalization_1/batchnorm/ReadVariableOp_16model/batch_normalization_1/batchnorm/ReadVariableOp_12p
6model/batch_normalization_1/batchnorm/ReadVariableOp_26model/batch_normalization_1/batchnorm/ReadVariableOp_22t
8model/batch_normalization_1/batchnorm/mul/ReadVariableOp8model/batch_normalization_1/batchnorm/mul/ReadVariableOp2J
#model/conv1d/BiasAdd/ReadVariableOp#model/conv1d/BiasAdd/ReadVariableOp2b
/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp/model/conv1d/conv1d/ExpandDims_1/ReadVariableOp2N
%model/conv1d_1/BiasAdd/ReadVariableOp%model/conv1d_1/BiasAdd/ReadVariableOp2f
1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp1model/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp2L
$model/dense_5/BiasAdd/ReadVariableOp$model/dense_5/BiasAdd/ReadVariableOp2J
#model/dense_5/MatMul/ReadVariableOp#model/dense_5/MatMul/ReadVariableOp2L
$model/dense_6/BiasAdd/ReadVariableOp$model/dense_6/BiasAdd/ReadVariableOp2J
#model/dense_6/MatMul/ReadVariableOp#model/dense_6/MatMul/ReadVariableOp2~
=model/token_and_position_embedding/embedding/embedding_lookup=model/token_and_position_embedding/embedding/embedding_lookup2
?model/token_and_position_embedding/embedding_1/embedding_lookup?model/token_and_position_embedding/embedding_1/embedding_lookup2
Hmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpHmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp2
Lmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpLmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp2
Hmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpHmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp2
Lmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpLmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp2¬
Tmodel/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpTmodel/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp2À
^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2
Gmodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOpGmodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOp2¦
Qmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpQmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2
Imodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpImodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOp2ª
Smodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpSmodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2
Imodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpImodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOp2ª
Smodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpSmodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2
Emodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpEmodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp2
Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpGmodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp2
Emodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpEmodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp2
Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpGmodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
è
§
4__inference_batch_normalization_layer_call_fn_141562

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1389612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¯ü
Ò
O__inference_transformer_block_1_layer_call_and_return_conditional_losses_141981

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
identity¢.layer_normalization_2/batchnorm/ReadVariableOp¢2layer_normalization_2/batchnorm/mul/ReadVariableOp¢.layer_normalization_3/batchnorm/ReadVariableOp¢2layer_normalization_3/batchnorm/mul/ReadVariableOp¢:multi_head_attention_1/attention_output/add/ReadVariableOp¢Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_1/key/add/ReadVariableOp¢7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_1/query/add/ReadVariableOp¢9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_1/value/add/ReadVariableOp¢9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp¢+sequential_1/dense_2/BiasAdd/ReadVariableOp¢-sequential_1/dense_2/Tensordot/ReadVariableOp¢+sequential_1/dense_3/BiasAdd/ReadVariableOp¢-sequential_1/dense_3/Tensordot/ReadVariableOpý
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_1/query/einsum/EinsumEinsuminputsAmulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/EinsumÛ
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/query/add/ReadVariableOpõ
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_1/query/add÷
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_1/key/einsum/EinsumEinsuminputs?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/EinsumÕ
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_1/key/add/ReadVariableOpí
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
multi_head_attention_1/key/addý
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_1/value/einsum/EinsumEinsuminputsAmulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/EinsumÛ
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/value/add/ReadVariableOpõ
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_1/value/add
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_1/Mul/yÆ
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_1/Mulü
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/EinsumÄ
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2(
&multi_head_attention_1/softmax/Softmax¡
,multi_head_attention_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,multi_head_attention_1/dropout/dropout/Const
*multi_head_attention_1/dropout/dropout/MulMul0multi_head_attention_1/softmax/Softmax:softmax:05multi_head_attention_1/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2,
*multi_head_attention_1/dropout/dropout/Mul¼
,multi_head_attention_1/dropout/dropout/ShapeShape0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_1/dropout/dropout/Shape
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_1/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
dtype02E
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform³
5multi_head_attention_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_1/dropout/dropout/GreaterEqual/yÂ
3multi_head_attention_1/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_1/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##25
3multi_head_attention_1/dropout/dropout/GreaterEqualä
+multi_head_attention_1/dropout/dropout/CastCast7multi_head_attention_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2-
+multi_head_attention_1/dropout/dropout/Castþ
,multi_head_attention_1/dropout/dropout/Mul_1Mul.multi_head_attention_1/dropout/dropout/Mul:z:0/multi_head_attention_1/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2.
,multi_head_attention_1/dropout/dropout/Mul_1
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/dropout/Mul_1:z:0$multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/Einsum
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/Einsumø
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOp
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+multi_head_attention_1/attention_output/addw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_2/dropout/Const¾
dropout_2/dropout/MulMul/multi_head_attention_1/attention_output/add:z:0 dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_2/dropout/Mul
dropout_2/dropout/ShapeShape/multi_head_attention_1/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_2/dropout/ShapeÖ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_2/dropout/GreaterEqual/yê
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
dropout_2/dropout/GreaterEqual¡
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_2/dropout/Cast¦
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_2/dropout/Mul_1n
addAddV2inputsdropout_2/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¶
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indicesß
"layer_normalization_2/moments/meanMeanadd:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2$
"layer_normalization_2/moments/meanË
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2,
*layer_normalization_2/moments/StopGradientë
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/layer_normalization_2/moments/SquaredDifference¾
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indices
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2(
&layer_normalization_2/moments/variance
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752'
%layer_normalization_2/batchnorm/add/yê
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#layer_normalization_2/batchnorm/add¶
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2'
%layer_normalization_2/batchnorm/Rsqrtà
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOpî
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_2/batchnorm/mul½
%layer_normalization_2/batchnorm/mul_1Muladd:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_2/batchnorm/mul_1á
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_2/batchnorm/mul_2Ô
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_2/batchnorm/ReadVariableOpê
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_2/batchnorm/subá
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_2/batchnorm/add_1Õ
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02/
-sequential_1/dense_2/Tensordot/ReadVariableOp
#sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_2/Tensordot/axes
#sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_2/Tensordot/free¥
$sequential_1/dense_2/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/Shape
,sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/GatherV2/axisº
'sequential_1/dense_2/Tensordot/GatherV2GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/free:output:05sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/GatherV2¢
.sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_2/Tensordot/GatherV2_1/axisÀ
)sequential_1/dense_2/Tensordot/GatherV2_1GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/axes:output:07sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_2/Tensordot/GatherV2_1
$sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_2/Tensordot/ConstÔ
#sequential_1/dense_2/Tensordot/ProdProd0sequential_1/dense_2/Tensordot/GatherV2:output:0-sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_2/Tensordot/Prod
&sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_1Ü
%sequential_1/dense_2/Tensordot/Prod_1Prod2sequential_1/dense_2/Tensordot/GatherV2_1:output:0/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_2/Tensordot/Prod_1
*sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_2/Tensordot/concat/axis
%sequential_1/dense_2/Tensordot/concatConcatV2,sequential_1/dense_2/Tensordot/free:output:0,sequential_1/dense_2/Tensordot/axes:output:03sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_2/Tensordot/concatà
$sequential_1/dense_2/Tensordot/stackPack,sequential_1/dense_2/Tensordot/Prod:output:0.sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/stackò
(sequential_1/dense_2/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0.sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2*
(sequential_1/dense_2/Tensordot/transposeó
&sequential_1/dense_2/Tensordot/ReshapeReshape,sequential_1/dense_2/Tensordot/transpose:y:0-sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&sequential_1/dense_2/Tensordot/Reshapeò
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%sequential_1/dense_2/Tensordot/MatMul
&sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2(
&sequential_1/dense_2/Tensordot/Const_2
,sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/concat_1/axis¦
'sequential_1/dense_2/Tensordot/concat_1ConcatV20sequential_1/dense_2/Tensordot/GatherV2:output:0/sequential_1/dense_2/Tensordot/Const_2:output:05sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/concat_1ä
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:00sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2 
sequential_1/dense_2/TensordotË
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOpÛ
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_1/dense_2/BiasAdd
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_1/dense_2/ReluÕ
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02/
-sequential_1/dense_3/Tensordot/ReadVariableOp
#sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_3/Tensordot/axes
#sequential_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_3/Tensordot/free£
$sequential_1/dense_3/Tensordot/ShapeShape'sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/Shape
,sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/GatherV2/axisº
'sequential_1/dense_3/Tensordot/GatherV2GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/free:output:05sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/GatherV2¢
.sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_3/Tensordot/GatherV2_1/axisÀ
)sequential_1/dense_3/Tensordot/GatherV2_1GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/axes:output:07sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_3/Tensordot/GatherV2_1
$sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_3/Tensordot/ConstÔ
#sequential_1/dense_3/Tensordot/ProdProd0sequential_1/dense_3/Tensordot/GatherV2:output:0-sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_3/Tensordot/Prod
&sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_1Ü
%sequential_1/dense_3/Tensordot/Prod_1Prod2sequential_1/dense_3/Tensordot/GatherV2_1:output:0/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_3/Tensordot/Prod_1
*sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_3/Tensordot/concat/axis
%sequential_1/dense_3/Tensordot/concatConcatV2,sequential_1/dense_3/Tensordot/free:output:0,sequential_1/dense_3/Tensordot/axes:output:03sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_3/Tensordot/concatà
$sequential_1/dense_3/Tensordot/stackPack,sequential_1/dense_3/Tensordot/Prod:output:0.sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/stackð
(sequential_1/dense_3/Tensordot/transpose	Transpose'sequential_1/dense_2/Relu:activations:0.sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2*
(sequential_1/dense_3/Tensordot/transposeó
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_3/Tensordot/transpose:y:0-sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&sequential_1/dense_3/Tensordot/Reshapeò
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential_1/dense_3/Tensordot/MatMul
&sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_2
,sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/concat_1/axis¦
'sequential_1/dense_3/Tensordot/concat_1ConcatV20sequential_1/dense_3/Tensordot/GatherV2:output:0/sequential_1/dense_3/Tensordot/Const_2:output:05sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/concat_1ä
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:00sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
sequential_1/dense_3/TensordotË
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOpÛ
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
sequential_1/dense_3/BiasAddw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_3/dropout/Const´
dropout_3/dropout/MulMul%sequential_1/dense_3/BiasAdd:output:0 dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_3/dropout/Mul
dropout_3/dropout/ShapeShape%sequential_1/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/ShapeÖ
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_3/dropout/GreaterEqual/yê
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
dropout_3/dropout/GreaterEqual¡
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_3/dropout/Cast¦
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_3/dropout/Mul_1
add_1AddV2)layer_normalization_2/batchnorm/add_1:z:0dropout_3/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¶
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indicesá
"layer_normalization_3/moments/meanMean	add_1:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2$
"layer_normalization_3/moments/meanË
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2,
*layer_normalization_3/moments/StopGradientí
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/layer_normalization_3/moments/SquaredDifference¾
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indices
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2(
&layer_normalization_3/moments/variance
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752'
%layer_normalization_3/batchnorm/add/yê
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#layer_normalization_3/batchnorm/add¶
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2'
%layer_normalization_3/batchnorm/Rsqrtà
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOpî
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_3/batchnorm/mul¿
%layer_normalization_3/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_3/batchnorm/mul_1á
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_3/batchnorm/mul_2Ô
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_3/batchnorm/ReadVariableOpê
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_3/batchnorm/subá
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_3/batchnorm/add_1Ó
IdentityIdentity)layer_normalization_3/batchnorm/add_1:z:0/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2
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
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ñ0
Æ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_138961

inputs
assignmovingavg_138936
assignmovingavg_1_138942)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/138936*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_138936*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/138936*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/138936*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_138936AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/138936*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/138942*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_138942*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/138942*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/138942*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_138942AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/138942*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
÷
k
O__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_138859

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDims¼
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize	
¬*
paddingVALID*
strides	
¬2	
AvgPool
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
Â(
__inference__traced_save_142784
file_prefix,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop-
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
 savev2_count_read_readvariableop9
5savev2_sgd_conv1d_kernel_momentum_read_readvariableop7
3savev2_sgd_conv1d_bias_momentum_read_readvariableop;
7savev2_sgd_conv1d_1_kernel_momentum_read_readvariableop9
5savev2_sgd_conv1d_1_bias_momentum_read_readvariableopE
Asavev2_sgd_batch_normalization_gamma_momentum_read_readvariableopD
@savev2_sgd_batch_normalization_beta_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_1_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_1_beta_momentum_read_readvariableop:
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

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameã%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*õ$
valueë$Bè$KB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¡
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*«
value¡BKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices²'
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableopLsavev2_token_and_position_embedding_embedding_embeddings_read_readvariableopNsavev2_token_and_position_embedding_embedding_1_embeddings_read_readvariableopRsavev2_transformer_block_1_multi_head_attention_1_query_kernel_read_readvariableopPsavev2_transformer_block_1_multi_head_attention_1_query_bias_read_readvariableopPsavev2_transformer_block_1_multi_head_attention_1_key_kernel_read_readvariableopNsavev2_transformer_block_1_multi_head_attention_1_key_bias_read_readvariableopRsavev2_transformer_block_1_multi_head_attention_1_value_kernel_read_readvariableopPsavev2_transformer_block_1_multi_head_attention_1_value_bias_read_readvariableop]savev2_transformer_block_1_multi_head_attention_1_attention_output_kernel_read_readvariableop[savev2_transformer_block_1_multi_head_attention_1_attention_output_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableopJsavev2_transformer_block_1_layer_normalization_2_gamma_read_readvariableopIsavev2_transformer_block_1_layer_normalization_2_beta_read_readvariableopJsavev2_transformer_block_1_layer_normalization_3_gamma_read_readvariableopIsavev2_transformer_block_1_layer_normalization_3_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_sgd_conv1d_kernel_momentum_read_readvariableop3savev2_sgd_conv1d_bias_momentum_read_readvariableop7savev2_sgd_conv1d_1_kernel_momentum_read_readvariableop5savev2_sgd_conv1d_1_bias_momentum_read_readvariableopAsavev2_sgd_batch_normalization_gamma_momentum_read_readvariableop@savev2_sgd_batch_normalization_beta_momentum_read_readvariableopCsavev2_sgd_batch_normalization_1_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_1_beta_momentum_read_readvariableop6savev2_sgd_dense_4_kernel_momentum_read_readvariableop4savev2_sgd_dense_4_bias_momentum_read_readvariableop6savev2_sgd_dense_5_kernel_momentum_read_readvariableop4savev2_sgd_dense_5_bias_momentum_read_readvariableop6savev2_sgd_dense_6_kernel_momentum_read_readvariableop4savev2_sgd_dense_6_bias_momentum_read_readvariableopYsavev2_sgd_token_and_position_embedding_embedding_embeddings_momentum_read_readvariableop[savev2_sgd_token_and_position_embedding_embedding_1_embeddings_momentum_read_readvariableop_savev2_sgd_transformer_block_1_multi_head_attention_1_query_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_1_multi_head_attention_1_query_bias_momentum_read_readvariableop]savev2_sgd_transformer_block_1_multi_head_attention_1_key_kernel_momentum_read_readvariableop[savev2_sgd_transformer_block_1_multi_head_attention_1_key_bias_momentum_read_readvariableop_savev2_sgd_transformer_block_1_multi_head_attention_1_value_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_1_multi_head_attention_1_value_bias_momentum_read_readvariableopjsavev2_sgd_transformer_block_1_multi_head_attention_1_attention_output_kernel_momentum_read_readvariableophsavev2_sgd_transformer_block_1_multi_head_attention_1_attention_output_bias_momentum_read_readvariableop6savev2_sgd_dense_2_kernel_momentum_read_readvariableop4savev2_sgd_dense_2_bias_momentum_read_readvariableop6savev2_sgd_dense_3_kernel_momentum_read_readvariableop4savev2_sgd_dense_3_bias_momentum_read_readvariableopWsavev2_sgd_transformer_block_1_layer_normalization_2_gamma_momentum_read_readvariableopVsavev2_sgd_transformer_block_1_layer_normalization_2_beta_momentum_read_readvariableopWsavev2_sgd_transformer_block_1_layer_normalization_3_gamma_momentum_read_readvariableopVsavev2_sgd_transformer_block_1_layer_normalization_3_beta_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Y
dtypesO
M2K	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*ñ
_input_shapesß
Ü: :  : :	  : : : : : : : : : :	è@:@:@@:@:@:: : : : : :	R :  : :  : :  : :  : : @:@:@ : : : : : : : :  : :	  : : : : : :	è@:@:@@:@:@:: :	R :  : :  : :  : :  : : @:@:@ : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:  : 

_output_shapes
: :($
"
_output_shapes
:	  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	è@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: :%!

_output_shapes
:	R :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :  

_output_shapes
: :$! 

_output_shapes

: @: "

_output_shapes
:@:$# 

_output_shapes

:@ : $

_output_shapes
: : %

_output_shapes
: : &

_output_shapes
: : '

_output_shapes
: : (

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :(+$
"
_output_shapes
:  : ,

_output_shapes
: :(-$
"
_output_shapes
:	  : .

_output_shapes
: : /

_output_shapes
: : 0

_output_shapes
: : 1

_output_shapes
: : 2

_output_shapes
: :%3!

_output_shapes
:	è@: 4

_output_shapes
:@:$5 

_output_shapes

:@@: 6

_output_shapes
:@:$7 

_output_shapes

:@: 8

_output_shapes
::$9 

_output_shapes

: :%:!

_output_shapes
:	R :(;$
"
_output_shapes
:  :$< 

_output_shapes

: :(=$
"
_output_shapes
:  :$> 

_output_shapes

: :(?$
"
_output_shapes
:  :$@ 

_output_shapes

: :(A$
"
_output_shapes
:  : B

_output_shapes
: :$C 

_output_shapes

: @: D

_output_shapes
:@:$E 

_output_shapes

:@ : F

_output_shapes
: : G

_output_shapes
: : H

_output_shapes
: : I

_output_shapes
: : J

_output_shapes
: :K

_output_shapes
: 

P
4__inference_average_pooling1d_1_layer_call_fn_138850

inputs
identityæ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_1388442
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼0
È
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_141775

inputs
assignmovingavg_141750
assignmovingavg_1_141756)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/141750*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_141750*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/141750*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/141750*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_141750AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/141750*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/141756*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_141756*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/141756*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/141756*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_141756AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/141756*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ó
i
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_138829

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDimsº
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
AvgPool
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_142243

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ñ0
Æ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_141529

inputs
assignmovingavg_141504
assignmovingavg_1_141510)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/141504*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_141504*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/141504*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/141504*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_141504AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/141504*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/141510*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_141510*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/141510*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/141510*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_141510AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/141510*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
í	
Ü
C__inference_dense_5_layer_call_and_return_conditional_losses_140103

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
º0
Æ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_139459

inputs
assignmovingavg_139434
assignmovingavg_1_139440)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/139434*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_139434*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/139434*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/139434*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_139434AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/139434*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/139440*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_139440*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/139440*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/139440*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_139440AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/139440*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
æ

O__inference_batch_normalization_layer_call_and_return_conditional_losses_141631

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

õ
B__inference_conv1d_layer_call_and_return_conditional_losses_141459

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿR ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 
 
_user_specified_nameinputs
ï
|
'__inference_conv1d_layer_call_fn_141468

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_1393732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿR ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 
 
_user_specified_nameinputs
Ý
}
(__inference_dense_6_layer_call_fn_142319

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1401592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¨
P
$__inference_add_layer_call_fn_141833
inputs_0
inputs_1
identityÑ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_1396122
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ# :ÿÿÿÿÿÿÿÿÿ# :U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
"
_user_specified_name
inputs/1
³
_
C__inference_flatten_layer_call_and_return_conditional_losses_140011

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ# :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
Ê
þ
X__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_141434
x'
#embedding_1_embedding_lookup_141421%
!embedding_embedding_lookup_141427
identity¢embedding/embedding_lookup¢embedding_1/embedding_lookup?
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
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice/stack_2â
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
range/delta
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
range¯
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_141421range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_1/embedding_lookup/141421*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02
embedding_1/embedding_lookup
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/141421*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%embedding_1/embedding_lookup/IdentityÀ
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'embedding_1/embedding_lookup/Identity_1m
embedding/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
embedding/Cast°
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_141427embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/141427*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02
embedding/embedding_lookup
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/141427*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2%
#embedding/embedding_lookup/Identity¿
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2'
%embedding/embedding_lookup/Identity_1¬
addAddV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
add
IdentityIdentityadd:z:0^embedding/embedding_lookup^embedding_1/embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿR::28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR

_user_specified_namex
Ý
}
(__inference_dense_5_layer_call_fn_142273

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1401032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_141713

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
í	
Ü
C__inference_dense_5_layer_call_and_return_conditional_losses_142264

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

d
E__inference_dropout_4_layer_call_and_return_conditional_losses_142238

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

d
E__inference_dropout_4_layer_call_and_return_conditional_losses_140074

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
â
Þ%
A__inference_model_layer_call_and_return_conditional_losses_141011
inputs_0
inputs_1D
@token_and_position_embedding_embedding_1_embedding_lookup_140713B
>token_and_position_embedding_embedding_embedding_lookup_1407196
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource.
*batch_normalization_assignmovingavg_1407690
,batch_normalization_assignmovingavg_1_140775=
9batch_normalization_batchnorm_mul_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource0
,batch_normalization_1_assignmovingavg_1408012
.batch_normalization_1_assignmovingavg_1_140807?
;batch_normalization_1_batchnorm_mul_readvariableop_resource;
7batch_normalization_1_batchnorm_readvariableop_resourceZ
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
identity¢7batch_normalization/AssignMovingAvg/AssignSubVariableOp¢2batch_normalization/AssignMovingAvg/ReadVariableOp¢9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp¢4batch_normalization/AssignMovingAvg_1/ReadVariableOp¢,batch_normalization/batchnorm/ReadVariableOp¢0batch_normalization/batchnorm/mul/ReadVariableOp¢9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp¢4batch_normalization_1/AssignMovingAvg/ReadVariableOp¢;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp¢6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_1/batchnorm/ReadVariableOp¢2batch_normalization_1/batchnorm/mul/ReadVariableOp¢conv1d/BiasAdd/ReadVariableOp¢)conv1d/conv1d/ExpandDims_1/ReadVariableOp¢conv1d_1/BiasAdd/ReadVariableOp¢+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢7token_and_position_embedding/embedding/embedding_lookup¢9token_and_position_embedding/embedding_1/embedding_lookup¢Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp¢Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp¢Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp¢Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp¢Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp¢Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp¢Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp¢Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp¢Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp¢Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp¢Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp¢Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp¢?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp¢Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp¢?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp¢Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp
"token_and_position_embedding/ShapeShapeinputs_0*
T0*
_output_shapes
:2$
"token_and_position_embedding/Shape·
0token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ22
0token_and_position_embedding/strided_slice/stack²
2token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 24
2token_and_position_embedding/strided_slice/stack_1²
2token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2token_and_position_embedding/strided_slice/stack_2
*token_and_position_embedding/strided_sliceStridedSlice+token_and_position_embedding/Shape:output:09token_and_position_embedding/strided_slice/stack:output:0;token_and_position_embedding/strided_slice/stack_1:output:0;token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*token_and_position_embedding/strided_slice
(token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2*
(token_and_position_embedding/range/start
(token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2*
(token_and_position_embedding/range/delta
"token_and_position_embedding/rangeRange1token_and_position_embedding/range/start:output:03token_and_position_embedding/strided_slice:output:01token_and_position_embedding/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"token_and_position_embedding/rangeÀ
9token_and_position_embedding/embedding_1/embedding_lookupResourceGather@token_and_position_embedding_embedding_1_embedding_lookup_140713+token_and_position_embedding/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*S
_classI
GEloc:@token_and_position_embedding/embedding_1/embedding_lookup/140713*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02;
9token_and_position_embedding/embedding_1/embedding_lookup
Btoken_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityBtoken_and_position_embedding/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*S
_classI
GEloc:@token_and_position_embedding/embedding_1/embedding_lookup/140713*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2D
Btoken_and_position_embedding/embedding_1/embedding_lookup/Identity
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityKtoken_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2F
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1®
+token_and_position_embedding/embedding/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2-
+token_and_position_embedding/embedding/CastÁ
7token_and_position_embedding/embedding/embedding_lookupResourceGather>token_and_position_embedding_embedding_embedding_lookup_140719/token_and_position_embedding/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@token_and_position_embedding/embedding/embedding_lookup/140719*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype029
7token_and_position_embedding/embedding/embedding_lookup
@token_and_position_embedding/embedding/embedding_lookup/IdentityIdentity@token_and_position_embedding/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@token_and_position_embedding/embedding/embedding_lookup/140719*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2B
@token_and_position_embedding/embedding/embedding_lookup/Identity
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityItoken_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2D
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1 
 token_and_position_embedding/addAddV2Ktoken_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Mtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2"
 token_and_position_embedding/add
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/conv1d/ExpandDims/dimÊ
conv1d/conv1d/ExpandDims
ExpandDims$token_and_position_embedding/add:z:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d/conv1d/ExpandDimsÍ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimÓ
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/conv1d/ExpandDims_1Ó
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
paddingSAME*
strides
2
conv1d/conv1d¨
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/conv1d/Squeeze¡
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1d/BiasAdd/ReadVariableOp©
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d/BiasAddr
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d/Relu
 average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 average_pooling1d/ExpandDims/dimË
average_pooling1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0)average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
average_pooling1d/ExpandDimsß
average_pooling1d/AvgPoolAvgPool%average_pooling1d/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
ksize
*
paddingVALID*
strides
2
average_pooling1d/AvgPool³
average_pooling1d/SqueezeSqueeze"average_pooling1d/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims
2
average_pooling1d/Squeeze
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_1/conv1d/ExpandDims/dimÎ
conv1d_1/conv1d/ExpandDims
ExpandDims"average_pooling1d/Squeeze:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_1/conv1d/ExpandDimsÓ
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dimÛ
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d_1/conv1d/ExpandDims_1Û
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
conv1d_1/conv1d®
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_1/conv1d/Squeeze§
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_1/BiasAdd/ReadVariableOp±
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_1/BiasAddx
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_1/Relu
"average_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_2/ExpandDims/dimÜ
average_pooling1d_2/ExpandDims
ExpandDims$token_and_position_embedding/add:z:0+average_pooling1d_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2 
average_pooling1d_2/ExpandDimsæ
average_pooling1d_2/AvgPoolAvgPool'average_pooling1d_2/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize	
¬*
paddingVALID*
strides	
¬2
average_pooling1d_2/AvgPool¸
average_pooling1d_2/SqueezeSqueeze$average_pooling1d_2/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_2/Squeeze
"average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_1/ExpandDims/dimÓ
average_pooling1d_1/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0+average_pooling1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2 
average_pooling1d_1/ExpandDimsä
average_pooling1d_1/AvgPoolAvgPool'average_pooling1d_1/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize

*
paddingVALID*
strides

2
average_pooling1d_1/AvgPool¸
average_pooling1d_1/SqueezeSqueeze$average_pooling1d_1/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_1/Squeeze¹
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indicesí
 batch_normalization/moments/meanMean$average_pooling1d_1/Squeeze:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2"
 batch_normalization/moments/mean¼
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
: 2*
(batch_normalization/moments/StopGradient
-batch_normalization/moments/SquaredDifferenceSquaredDifference$average_pooling1d_1/Squeeze:output:01batch_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2/
-batch_normalization/moments/SquaredDifferenceÁ
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       28
6batch_normalization/moments/variance/reduction_indices
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2&
$batch_normalization/moments/variance½
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2%
#batch_normalization/moments/SqueezeÅ
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1
)batch_normalization/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/140769*
_output_shapes
: *
dtype0*
valueB
 *
×#<2+
)batch_normalization/AssignMovingAvg/decayÏ
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_assignmovingavg_140769*
_output_shapes
: *
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpÕ
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/140769*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/subÌ
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/140769*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/mul§
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_assignmovingavg_140769+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@batch_normalization/AssignMovingAvg/140769*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOp
+batch_normalization/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/140775*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization/AssignMovingAvg_1/decayÕ
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_assignmovingavg_1_140775*
_output_shapes
: *
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOpß
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/140775*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/subÖ
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/140775*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/mul³
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_assignmovingavg_1_140775-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization/AssignMovingAvg_1/140775*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2%
#batch_normalization/batchnorm/add/yÒ
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/add
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/RsqrtÚ
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpÕ
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/mulÔ
#batch_normalization/batchnorm/mul_1Mul$average_pooling1d_1/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#batch_normalization/batchnorm/mul_1Ë
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/mul_2Î
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02.
,batch_normalization/batchnorm/ReadVariableOpÑ
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/subÙ
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#batch_normalization/batchnorm/add_1½
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indicesó
"batch_normalization_1/moments/meanMean$average_pooling1d_2/Squeeze:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_1/moments/meanÂ
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_1/moments/StopGradient
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference$average_pooling1d_2/Squeeze:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/batch_normalization_1/moments/SquaredDifferenceÅ
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_1/moments/variance/reduction_indices
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2(
&batch_normalization_1/moments/varianceÃ
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_1/moments/SqueezeË
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1
+batch_normalization_1/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/140801*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_1/AssignMovingAvg/decayÕ
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_1_assignmovingavg_140801*
_output_shapes
: *
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOpß
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/140801*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/subÖ
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/140801*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/mul³
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_1_assignmovingavg_140801-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg/140801*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp
-batch_normalization_1/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/140807*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_1/AssignMovingAvg_1/decayÛ
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_1_assignmovingavg_1_140807*
_output_shapes
: *
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpé
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/140807*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/subà
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/140807*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/mul¿
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_1_assignmovingavg_1_140807/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_1/AssignMovingAvg_1/140807*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_1/batchnorm/add/yÚ
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/add¥
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_1/batchnorm/Rsqrtà
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/mulÚ
%batch_normalization_1/batchnorm/mul_1Mul$average_pooling1d_2/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_1/batchnorm/mul_1Ó
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_1/batchnorm/mul_2Ô
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_1/batchnorm/ReadVariableOpÙ
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/subá
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_1/batchnorm/add_1¥
add/addAddV2'batch_normalization/batchnorm/add_1:z:0)batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2	
add/add¹
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_1_multi_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpÎ
>transformer_block_1/multi_head_attention_1/query/einsum/EinsumEinsumadd/add:z:0Utransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_1/multi_head_attention_1/query/einsum/Einsum
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOpReadVariableOpLtransformer_block_1_multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOpÅ
4transformer_block_1/multi_head_attention_1/query/addAddV2Gtransformer_block_1/multi_head_attention_1/query/einsum/Einsum:output:0Ktransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_1/multi_head_attention_1/query/add³
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_1_multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpÈ
<transformer_block_1/multi_head_attention_1/key/einsum/EinsumEinsumadd/add:z:0Stransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2>
<transformer_block_1/multi_head_attention_1/key/einsum/Einsum
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOpReadVariableOpJtransformer_block_1_multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp½
2transformer_block_1/multi_head_attention_1/key/addAddV2Etransformer_block_1/multi_head_attention_1/key/einsum/Einsum:output:0Itransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 24
2transformer_block_1/multi_head_attention_1/key/add¹
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_1_multi_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpÎ
>transformer_block_1/multi_head_attention_1/value/einsum/EinsumEinsumadd/add:z:0Utransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_1/multi_head_attention_1/value/einsum/Einsum
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOpReadVariableOpLtransformer_block_1_multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOpÅ
4transformer_block_1/multi_head_attention_1/value/addAddV2Gtransformer_block_1/multi_head_attention_1/value/einsum/Einsum:output:0Ktransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_1/multi_head_attention_1/value/add©
0transformer_block_1/multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>22
0transformer_block_1/multi_head_attention_1/Mul/y
.transformer_block_1/multi_head_attention_1/MulMul8transformer_block_1/multi_head_attention_1/query/add:z:09transformer_block_1/multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 20
.transformer_block_1/multi_head_attention_1/MulÌ
8transformer_block_1/multi_head_attention_1/einsum/EinsumEinsum6transformer_block_1/multi_head_attention_1/key/add:z:02transformer_block_1/multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2:
8transformer_block_1/multi_head_attention_1/einsum/Einsum
:transformer_block_1/multi_head_attention_1/softmax/SoftmaxSoftmaxAtransformer_block_1/multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2<
:transformer_block_1/multi_head_attention_1/softmax/SoftmaxÉ
@transformer_block_1/multi_head_attention_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2B
@transformer_block_1/multi_head_attention_1/dropout/dropout/ConstÒ
>transformer_block_1/multi_head_attention_1/dropout/dropout/MulMulDtransformer_block_1/multi_head_attention_1/softmax/Softmax:softmax:0Itransformer_block_1/multi_head_attention_1/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2@
>transformer_block_1/multi_head_attention_1/dropout/dropout/Mulø
@transformer_block_1/multi_head_attention_1/dropout/dropout/ShapeShapeDtransformer_block_1/multi_head_attention_1/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2B
@transformer_block_1/multi_head_attention_1/dropout/dropout/ShapeÕ
Wtransformer_block_1/multi_head_attention_1/dropout/dropout/random_uniform/RandomUniformRandomUniformItransformer_block_1/multi_head_attention_1/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
dtype02Y
Wtransformer_block_1/multi_head_attention_1/dropout/dropout/random_uniform/RandomUniformÛ
Itransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Itransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqual/y
Gtransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqualGreaterEqual`transformer_block_1/multi_head_attention_1/dropout/dropout/random_uniform/RandomUniform:output:0Rtransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2I
Gtransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqual 
?transformer_block_1/multi_head_attention_1/dropout/dropout/CastCastKtransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2A
?transformer_block_1/multi_head_attention_1/dropout/dropout/CastÎ
@transformer_block_1/multi_head_attention_1/dropout/dropout/Mul_1MulBtransformer_block_1/multi_head_attention_1/dropout/dropout/Mul:z:0Ctransformer_block_1/multi_head_attention_1/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2B
@transformer_block_1/multi_head_attention_1/dropout/dropout/Mul_1ä
:transformer_block_1/multi_head_attention_1/einsum_1/EinsumEinsumDtransformer_block_1/multi_head_attention_1/dropout/dropout/Mul_1:z:08transformer_block_1/multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2<
:transformer_block_1/multi_head_attention_1/einsum_1/EinsumÚ
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_1_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp£
Itransformer_block_1/multi_head_attention_1/attention_output/einsum/EinsumEinsumCtransformer_block_1/multi_head_attention_1/einsum_1/Einsum:output:0`transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe2K
Itransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum´
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_1_multi_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpí
?transformer_block_1/multi_head_attention_1/attention_output/addAddV2Rtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum:output:0Vtransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?transformer_block_1/multi_head_attention_1/attention_output/add
+transformer_block_1/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2-
+transformer_block_1/dropout_2/dropout/Const
)transformer_block_1/dropout_2/dropout/MulMulCtransformer_block_1/multi_head_attention_1/attention_output/add:z:04transformer_block_1/dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2+
)transformer_block_1/dropout_2/dropout/MulÍ
+transformer_block_1/dropout_2/dropout/ShapeShapeCtransformer_block_1/multi_head_attention_1/attention_output/add:z:0*
T0*
_output_shapes
:2-
+transformer_block_1/dropout_2/dropout/Shape
Btransformer_block_1/dropout_2/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_1/dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype02D
Btransformer_block_1/dropout_2/dropout/random_uniform/RandomUniform±
4transformer_block_1/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=26
4transformer_block_1/dropout_2/dropout/GreaterEqual/yº
2transformer_block_1/dropout_2/dropout/GreaterEqualGreaterEqualKtransformer_block_1/dropout_2/dropout/random_uniform/RandomUniform:output:0=transformer_block_1/dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 24
2transformer_block_1/dropout_2/dropout/GreaterEqualÝ
*transformer_block_1/dropout_2/dropout/CastCast6transformer_block_1/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2,
*transformer_block_1/dropout_2/dropout/Castö
+transformer_block_1/dropout_2/dropout/Mul_1Mul-transformer_block_1/dropout_2/dropout/Mul:z:0.transformer_block_1/dropout_2/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+transformer_block_1/dropout_2/dropout/Mul_1¯
transformer_block_1/addAddV2add/add:z:0/transformer_block_1/dropout_2/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_1/addÞ
Htransformer_block_1/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_1/layer_normalization_2/moments/mean/reduction_indices¯
6transformer_block_1/layer_normalization_2/moments/meanMeantransformer_block_1/add:z:0Qtransformer_block_1/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(28
6transformer_block_1/layer_normalization_2/moments/mean
>transformer_block_1/layer_normalization_2/moments/StopGradientStopGradient?transformer_block_1/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2@
>transformer_block_1/layer_normalization_2/moments/StopGradient»
Ctransformer_block_1/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencetransformer_block_1/add:z:0Gtransformer_block_1/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2E
Ctransformer_block_1/layer_normalization_2/moments/SquaredDifferenceæ
Ltransformer_block_1/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_1/layer_normalization_2/moments/variance/reduction_indicesç
:transformer_block_1/layer_normalization_2/moments/varianceMeanGtransformer_block_1/layer_normalization_2/moments/SquaredDifference:z:0Utransformer_block_1/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2<
:transformer_block_1/layer_normalization_2/moments/variance»
9transformer_block_1/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752;
9transformer_block_1/layer_normalization_2/batchnorm/add/yº
7transformer_block_1/layer_normalization_2/batchnorm/addAddV2Ctransformer_block_1/layer_normalization_2/moments/variance:output:0Btransformer_block_1/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#29
7transformer_block_1/layer_normalization_2/batchnorm/addò
9transformer_block_1/layer_normalization_2/batchnorm/RsqrtRsqrt;transformer_block_1/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2;
9transformer_block_1/layer_normalization_2/batchnorm/Rsqrt
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp¾
7transformer_block_1/layer_normalization_2/batchnorm/mulMul=transformer_block_1/layer_normalization_2/batchnorm/Rsqrt:y:0Ntransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 29
7transformer_block_1/layer_normalization_2/batchnorm/mul
9transformer_block_1/layer_normalization_2/batchnorm/mul_1Multransformer_block_1/add:z:0;transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_1/layer_normalization_2/batchnorm/mul_1±
9transformer_block_1/layer_normalization_2/batchnorm/mul_2Mul?transformer_block_1/layer_normalization_2/moments/mean:output:0;transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_1/layer_normalization_2/batchnorm/mul_2
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_1_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpº
7transformer_block_1/layer_normalization_2/batchnorm/subSubJtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp:value:0=transformer_block_1/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 29
7transformer_block_1/layer_normalization_2/batchnorm/sub±
9transformer_block_1/layer_normalization_2/batchnorm/add_1AddV2=transformer_block_1/layer_normalization_2/batchnorm/mul_1:z:0;transformer_block_1/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_1/layer_normalization_2/batchnorm/add_1
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_1_sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02C
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp¼
7transformer_block_1/sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_1/sequential_1/dense_2/Tensordot/axesÃ
7transformer_block_1/sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_1/sequential_1/dense_2/Tensordot/freeá
8transformer_block_1/sequential_1/dense_2/Tensordot/ShapeShape=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_2/Tensordot/ShapeÆ
@transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axis
;transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2GatherV2Atransformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2Ê
Btransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axis¤
=transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1GatherV2Atransformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Ktransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1¾
8transformer_block_1/sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_1/sequential_1/dense_2/Tensordot/Const¤
7transformer_block_1/sequential_1/dense_2/Tensordot/ProdProdDtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Atransformer_block_1/sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_1/sequential_1/dense_2/Tensordot/ProdÂ
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_1¬
9transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1ProdFtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1:output:0Ctransformer_block_1/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1Â
>transformer_block_1/sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_1/sequential_1/dense_2/Tensordot/concat/axisý
9transformer_block_1/sequential_1/dense_2/Tensordot/concatConcatV2@transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Gtransformer_block_1/sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_1/sequential_1/dense_2/Tensordot/concat°
8transformer_block_1/sequential_1/dense_2/Tensordot/stackPack@transformer_block_1/sequential_1/dense_2/Tensordot/Prod:output:0Btransformer_block_1/sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_2/Tensordot/stackÂ
<transformer_block_1/sequential_1/dense_2/Tensordot/transpose	Transpose=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0Btransformer_block_1/sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2>
<transformer_block_1/sequential_1/dense_2/Tensordot/transposeÃ
:transformer_block_1/sequential_1/dense_2/Tensordot/ReshapeReshape@transformer_block_1/sequential_1/dense_2/Tensordot/transpose:y:0Atransformer_block_1/sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2<
:transformer_block_1/sequential_1/dense_2/Tensordot/ReshapeÂ
9transformer_block_1/sequential_1/dense_2/Tensordot/MatMulMatMulCtransformer_block_1/sequential_1/dense_2/Tensordot/Reshape:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2;
9transformer_block_1/sequential_1/dense_2/Tensordot/MatMulÂ
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2<
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_2Æ
@transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axis
;transformer_block_1/sequential_1/dense_2/Tensordot/concat_1ConcatV2Dtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Ctransformer_block_1/sequential_1/dense_2/Tensordot/Const_2:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_2/Tensordot/concat_1´
2transformer_block_1/sequential_1/dense_2/TensordotReshapeCtransformer_block_1/sequential_1/dense_2/Tensordot/MatMul:product:0Dtransformer_block_1/sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@24
2transformer_block_1/sequential_1/dense_2/Tensordot
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_1_sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02A
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp«
0transformer_block_1/sequential_1/dense_2/BiasAddBiasAdd;transformer_block_1/sequential_1/dense_2/Tensordot:output:0Gtransformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@22
0transformer_block_1/sequential_1/dense_2/BiasAdd×
-transformer_block_1/sequential_1/dense_2/ReluRelu9transformer_block_1/sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2/
-transformer_block_1/sequential_1/dense_2/Relu
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_1_sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02C
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp¼
7transformer_block_1/sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_1/sequential_1/dense_3/Tensordot/axesÃ
7transformer_block_1/sequential_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_1/sequential_1/dense_3/Tensordot/freeß
8transformer_block_1/sequential_1/dense_3/Tensordot/ShapeShape;transformer_block_1/sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_3/Tensordot/ShapeÆ
@transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axis
;transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2GatherV2Atransformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2Ê
Btransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axis¤
=transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1GatherV2Atransformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Ktransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1¾
8transformer_block_1/sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_1/sequential_1/dense_3/Tensordot/Const¤
7transformer_block_1/sequential_1/dense_3/Tensordot/ProdProdDtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Atransformer_block_1/sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_1/sequential_1/dense_3/Tensordot/ProdÂ
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_1¬
9transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1ProdFtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1:output:0Ctransformer_block_1/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1Â
>transformer_block_1/sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_1/sequential_1/dense_3/Tensordot/concat/axisý
9transformer_block_1/sequential_1/dense_3/Tensordot/concatConcatV2@transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Gtransformer_block_1/sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_1/sequential_1/dense_3/Tensordot/concat°
8transformer_block_1/sequential_1/dense_3/Tensordot/stackPack@transformer_block_1/sequential_1/dense_3/Tensordot/Prod:output:0Btransformer_block_1/sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_3/Tensordot/stackÀ
<transformer_block_1/sequential_1/dense_3/Tensordot/transpose	Transpose;transformer_block_1/sequential_1/dense_2/Relu:activations:0Btransformer_block_1/sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2>
<transformer_block_1/sequential_1/dense_3/Tensordot/transposeÃ
:transformer_block_1/sequential_1/dense_3/Tensordot/ReshapeReshape@transformer_block_1/sequential_1/dense_3/Tensordot/transpose:y:0Atransformer_block_1/sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2<
:transformer_block_1/sequential_1/dense_3/Tensordot/ReshapeÂ
9transformer_block_1/sequential_1/dense_3/Tensordot/MatMulMatMulCtransformer_block_1/sequential_1/dense_3/Tensordot/Reshape:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2;
9transformer_block_1/sequential_1/dense_3/Tensordot/MatMulÂ
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_2Æ
@transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axis
;transformer_block_1/sequential_1/dense_3/Tensordot/concat_1ConcatV2Dtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Ctransformer_block_1/sequential_1/dense_3/Tensordot/Const_2:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_3/Tensordot/concat_1´
2transformer_block_1/sequential_1/dense_3/TensordotReshapeCtransformer_block_1/sequential_1/dense_3/Tensordot/MatMul:product:0Dtransformer_block_1/sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 24
2transformer_block_1/sequential_1/dense_3/Tensordot
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_1_sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp«
0transformer_block_1/sequential_1/dense_3/BiasAddBiasAdd;transformer_block_1/sequential_1/dense_3/Tensordot:output:0Gtransformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0transformer_block_1/sequential_1/dense_3/BiasAdd
+transformer_block_1/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2-
+transformer_block_1/dropout_3/dropout/Const
)transformer_block_1/dropout_3/dropout/MulMul9transformer_block_1/sequential_1/dense_3/BiasAdd:output:04transformer_block_1/dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2+
)transformer_block_1/dropout_3/dropout/MulÃ
+transformer_block_1/dropout_3/dropout/ShapeShape9transformer_block_1/sequential_1/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2-
+transformer_block_1/dropout_3/dropout/Shape
Btransformer_block_1/dropout_3/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_1/dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype02D
Btransformer_block_1/dropout_3/dropout/random_uniform/RandomUniform±
4transformer_block_1/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=26
4transformer_block_1/dropout_3/dropout/GreaterEqual/yº
2transformer_block_1/dropout_3/dropout/GreaterEqualGreaterEqualKtransformer_block_1/dropout_3/dropout/random_uniform/RandomUniform:output:0=transformer_block_1/dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 24
2transformer_block_1/dropout_3/dropout/GreaterEqualÝ
*transformer_block_1/dropout_3/dropout/CastCast6transformer_block_1/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2,
*transformer_block_1/dropout_3/dropout/Castö
+transformer_block_1/dropout_3/dropout/Mul_1Mul-transformer_block_1/dropout_3/dropout/Mul:z:0.transformer_block_1/dropout_3/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+transformer_block_1/dropout_3/dropout/Mul_1å
transformer_block_1/add_1AddV2=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0/transformer_block_1/dropout_3/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_1/add_1Þ
Htransformer_block_1/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_1/layer_normalization_3/moments/mean/reduction_indices±
6transformer_block_1/layer_normalization_3/moments/meanMeantransformer_block_1/add_1:z:0Qtransformer_block_1/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(28
6transformer_block_1/layer_normalization_3/moments/mean
>transformer_block_1/layer_normalization_3/moments/StopGradientStopGradient?transformer_block_1/layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2@
>transformer_block_1/layer_normalization_3/moments/StopGradient½
Ctransformer_block_1/layer_normalization_3/moments/SquaredDifferenceSquaredDifferencetransformer_block_1/add_1:z:0Gtransformer_block_1/layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2E
Ctransformer_block_1/layer_normalization_3/moments/SquaredDifferenceæ
Ltransformer_block_1/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_1/layer_normalization_3/moments/variance/reduction_indicesç
:transformer_block_1/layer_normalization_3/moments/varianceMeanGtransformer_block_1/layer_normalization_3/moments/SquaredDifference:z:0Utransformer_block_1/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2<
:transformer_block_1/layer_normalization_3/moments/variance»
9transformer_block_1/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752;
9transformer_block_1/layer_normalization_3/batchnorm/add/yº
7transformer_block_1/layer_normalization_3/batchnorm/addAddV2Ctransformer_block_1/layer_normalization_3/moments/variance:output:0Btransformer_block_1/layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#29
7transformer_block_1/layer_normalization_3/batchnorm/addò
9transformer_block_1/layer_normalization_3/batchnorm/RsqrtRsqrt;transformer_block_1/layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2;
9transformer_block_1/layer_normalization_3/batchnorm/Rsqrt
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp¾
7transformer_block_1/layer_normalization_3/batchnorm/mulMul=transformer_block_1/layer_normalization_3/batchnorm/Rsqrt:y:0Ntransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 29
7transformer_block_1/layer_normalization_3/batchnorm/mul
9transformer_block_1/layer_normalization_3/batchnorm/mul_1Multransformer_block_1/add_1:z:0;transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_1/layer_normalization_3/batchnorm/mul_1±
9transformer_block_1/layer_normalization_3/batchnorm/mul_2Mul?transformer_block_1/layer_normalization_3/moments/mean:output:0;transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_1/layer_normalization_3/batchnorm/mul_2
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpº
7transformer_block_1/layer_normalization_3/batchnorm/subSubJtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp:value:0=transformer_block_1/layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 29
7transformer_block_1/layer_normalization_3/batchnorm/sub±
9transformer_block_1/layer_normalization_3/batchnorm/add_1AddV2=transformer_block_1/layer_normalization_3/batchnorm/mul_1:z:0;transformer_block_1/layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_1/layer_normalization_3/batchnorm/add_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
flatten/Const·
flatten/ReshapeReshape=transformer_block_1/layer_normalization_3/batchnorm/add_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
flatten/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis¶
concatenate/concatConcatV2flatten/Reshape:output:0inputs_1 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
concatenate/concat¦
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02
dense_4/MatMul/ReadVariableOp 
dense_4/MatMulMatMulconcatenate/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_4/MatMul¤
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_4/BiasAdd/ReadVariableOp¡
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_4/Reluw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_4/dropout/Const¥
dropout_4/dropout/MulMuldense_4/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_4/dropout/Mul|
dropout_4/dropout/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
:2
dropout_4/dropout/ShapeÒ
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_4/dropout/GreaterEqual/yæ
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
dropout_4/dropout/GreaterEqual
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_4/dropout/Cast¢
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_4/dropout/Mul_1¥
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_5/MatMul/ReadVariableOp 
dense_5/MatMulMatMuldropout_4/dropout/Mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_5/MatMul¤
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_5/BiasAdd/ReadVariableOp¡
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_5/Reluw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_5/dropout/Const¥
dropout_5/dropout/MulMuldense_5/Relu:activations:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_5/dropout/Mul|
dropout_5/dropout/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:2
dropout_5/dropout/ShapeÒ
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype020
.dropout_5/dropout/random_uniform/RandomUniform
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_5/dropout/GreaterEqual/yæ
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
dropout_5/dropout/GreaterEqual
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_5/dropout/Cast¢
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_5/dropout/Mul_1¥
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_6/MatMul/ReadVariableOp 
dense_6/MatMulMatMuldropout_5/dropout/Mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/MatMul¤
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp¡
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/BiasAddç
IdentityIdentitydense_6/BiasAdd:output:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp3^batch_normalization/AssignMovingAvg/ReadVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp5^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_1/AssignMovingAvg/ReadVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp8^token_and_position_embedding/embedding/embedding_lookup:^token_and_position_embedding/embedding_1/embedding_lookupC^transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpG^transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpC^transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpG^transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpO^transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpY^transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_1/multi_head_attention_1/key/add/ReadVariableOpL^transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpD^transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpN^transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpD^transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpN^transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp@^transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpB^transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp@^transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpB^transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2r
7token_and_position_embedding/embedding/embedding_lookup7token_and_position_embedding/embedding/embedding_lookup2v
9token_and_position_embedding/embedding_1/embedding_lookup9token_and_position_embedding/embedding_1/embedding_lookup2
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpBtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp2
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpFtransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp2
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpBtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp2
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpFtransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp2 
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpNtransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp2´
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOpAtransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp2
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpKtransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOpCtransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp2
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpMtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOpCtransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp2
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpMtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp2
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpAtransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp2
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp2
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpAtransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
£
c
*__inference_dropout_4_layer_call_fn_142248

inputs
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1400742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ü
Ò
O__inference_transformer_block_1_layer_call_and_return_conditional_losses_142108

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
identity¢.layer_normalization_2/batchnorm/ReadVariableOp¢2layer_normalization_2/batchnorm/mul/ReadVariableOp¢.layer_normalization_3/batchnorm/ReadVariableOp¢2layer_normalization_3/batchnorm/mul/ReadVariableOp¢:multi_head_attention_1/attention_output/add/ReadVariableOp¢Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_1/key/add/ReadVariableOp¢7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_1/query/add/ReadVariableOp¢9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_1/value/add/ReadVariableOp¢9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp¢+sequential_1/dense_2/BiasAdd/ReadVariableOp¢-sequential_1/dense_2/Tensordot/ReadVariableOp¢+sequential_1/dense_3/BiasAdd/ReadVariableOp¢-sequential_1/dense_3/Tensordot/ReadVariableOpý
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_1/query/einsum/EinsumEinsuminputsAmulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/EinsumÛ
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/query/add/ReadVariableOpõ
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_1/query/add÷
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_1/key/einsum/EinsumEinsuminputs?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/EinsumÕ
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_1/key/add/ReadVariableOpí
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
multi_head_attention_1/key/addý
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_1/value/einsum/EinsumEinsuminputsAmulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/EinsumÛ
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/value/add/ReadVariableOpõ
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_1/value/add
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_1/Mul/yÆ
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_1/Mulü
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/EinsumÄ
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2(
&multi_head_attention_1/softmax/SoftmaxÊ
'multi_head_attention_1/dropout/IdentityIdentity0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2)
'multi_head_attention_1/dropout/Identity
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/Identity:output:0$multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/Einsum
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/Einsumø
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOp
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+multi_head_attention_1/attention_output/add
dropout_2/IdentityIdentity/multi_head_attention_1/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_2/Identityn
addAddV2inputsdropout_2/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¶
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indicesß
"layer_normalization_2/moments/meanMeanadd:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2$
"layer_normalization_2/moments/meanË
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2,
*layer_normalization_2/moments/StopGradientë
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/layer_normalization_2/moments/SquaredDifference¾
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indices
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2(
&layer_normalization_2/moments/variance
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752'
%layer_normalization_2/batchnorm/add/yê
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#layer_normalization_2/batchnorm/add¶
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2'
%layer_normalization_2/batchnorm/Rsqrtà
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOpî
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_2/batchnorm/mul½
%layer_normalization_2/batchnorm/mul_1Muladd:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_2/batchnorm/mul_1á
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_2/batchnorm/mul_2Ô
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_2/batchnorm/ReadVariableOpê
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_2/batchnorm/subá
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_2/batchnorm/add_1Õ
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02/
-sequential_1/dense_2/Tensordot/ReadVariableOp
#sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_2/Tensordot/axes
#sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_2/Tensordot/free¥
$sequential_1/dense_2/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/Shape
,sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/GatherV2/axisº
'sequential_1/dense_2/Tensordot/GatherV2GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/free:output:05sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/GatherV2¢
.sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_2/Tensordot/GatherV2_1/axisÀ
)sequential_1/dense_2/Tensordot/GatherV2_1GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/axes:output:07sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_2/Tensordot/GatherV2_1
$sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_2/Tensordot/ConstÔ
#sequential_1/dense_2/Tensordot/ProdProd0sequential_1/dense_2/Tensordot/GatherV2:output:0-sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_2/Tensordot/Prod
&sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_1Ü
%sequential_1/dense_2/Tensordot/Prod_1Prod2sequential_1/dense_2/Tensordot/GatherV2_1:output:0/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_2/Tensordot/Prod_1
*sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_2/Tensordot/concat/axis
%sequential_1/dense_2/Tensordot/concatConcatV2,sequential_1/dense_2/Tensordot/free:output:0,sequential_1/dense_2/Tensordot/axes:output:03sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_2/Tensordot/concatà
$sequential_1/dense_2/Tensordot/stackPack,sequential_1/dense_2/Tensordot/Prod:output:0.sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/stackò
(sequential_1/dense_2/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0.sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2*
(sequential_1/dense_2/Tensordot/transposeó
&sequential_1/dense_2/Tensordot/ReshapeReshape,sequential_1/dense_2/Tensordot/transpose:y:0-sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&sequential_1/dense_2/Tensordot/Reshapeò
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%sequential_1/dense_2/Tensordot/MatMul
&sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2(
&sequential_1/dense_2/Tensordot/Const_2
,sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/concat_1/axis¦
'sequential_1/dense_2/Tensordot/concat_1ConcatV20sequential_1/dense_2/Tensordot/GatherV2:output:0/sequential_1/dense_2/Tensordot/Const_2:output:05sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/concat_1ä
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:00sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2 
sequential_1/dense_2/TensordotË
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOpÛ
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_1/dense_2/BiasAdd
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_1/dense_2/ReluÕ
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02/
-sequential_1/dense_3/Tensordot/ReadVariableOp
#sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_3/Tensordot/axes
#sequential_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_3/Tensordot/free£
$sequential_1/dense_3/Tensordot/ShapeShape'sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/Shape
,sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/GatherV2/axisº
'sequential_1/dense_3/Tensordot/GatherV2GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/free:output:05sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/GatherV2¢
.sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_3/Tensordot/GatherV2_1/axisÀ
)sequential_1/dense_3/Tensordot/GatherV2_1GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/axes:output:07sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_3/Tensordot/GatherV2_1
$sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_3/Tensordot/ConstÔ
#sequential_1/dense_3/Tensordot/ProdProd0sequential_1/dense_3/Tensordot/GatherV2:output:0-sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_3/Tensordot/Prod
&sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_1Ü
%sequential_1/dense_3/Tensordot/Prod_1Prod2sequential_1/dense_3/Tensordot/GatherV2_1:output:0/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_3/Tensordot/Prod_1
*sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_3/Tensordot/concat/axis
%sequential_1/dense_3/Tensordot/concatConcatV2,sequential_1/dense_3/Tensordot/free:output:0,sequential_1/dense_3/Tensordot/axes:output:03sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_3/Tensordot/concatà
$sequential_1/dense_3/Tensordot/stackPack,sequential_1/dense_3/Tensordot/Prod:output:0.sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/stackð
(sequential_1/dense_3/Tensordot/transpose	Transpose'sequential_1/dense_2/Relu:activations:0.sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2*
(sequential_1/dense_3/Tensordot/transposeó
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_3/Tensordot/transpose:y:0-sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&sequential_1/dense_3/Tensordot/Reshapeò
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential_1/dense_3/Tensordot/MatMul
&sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_2
,sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/concat_1/axis¦
'sequential_1/dense_3/Tensordot/concat_1ConcatV20sequential_1/dense_3/Tensordot/GatherV2:output:0/sequential_1/dense_3/Tensordot/Const_2:output:05sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/concat_1ä
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:00sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
sequential_1/dense_3/TensordotË
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOpÛ
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
sequential_1/dense_3/BiasAdd
dropout_3/IdentityIdentity%sequential_1/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_3/Identity
add_1AddV2)layer_normalization_2/batchnorm/add_1:z:0dropout_3/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¶
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indicesá
"layer_normalization_3/moments/meanMean	add_1:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2$
"layer_normalization_3/moments/meanË
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2,
*layer_normalization_3/moments/StopGradientí
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/layer_normalization_3/moments/SquaredDifference¾
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indices
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2(
&layer_normalization_3/moments/variance
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752'
%layer_normalization_3/batchnorm/add/yê
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#layer_normalization_3/batchnorm/add¶
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2'
%layer_normalization_3/batchnorm/Rsqrtà
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOpî
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_3/batchnorm/mul¿
%layer_normalization_3/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_3/batchnorm/mul_1á
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_3/batchnorm/mul_2Ô
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_3/batchnorm/ReadVariableOpê
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_3/batchnorm/subá
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_3/batchnorm/add_1Ó
IdentityIdentity)layer_normalization_3/batchnorm/add_1:z:0/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2
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
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
£
c
*__inference_dropout_5_layer_call_fn_142295

inputs
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1401312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

F
*__inference_dropout_5_layer_call_fn_142300

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1401362
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


=__inference_token_and_position_embedding_layer_call_fn_141443
x
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_1393412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿR::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR

_user_specified_namex
Ö
¢
&__inference_model_layer_call_fn_141410
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

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity¢StatefulPartitionedCallÔ
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1405402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ü
Ò
O__inference_transformer_block_1_layer_call_and_return_conditional_losses_139896

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
identity¢.layer_normalization_2/batchnorm/ReadVariableOp¢2layer_normalization_2/batchnorm/mul/ReadVariableOp¢.layer_normalization_3/batchnorm/ReadVariableOp¢2layer_normalization_3/batchnorm/mul/ReadVariableOp¢:multi_head_attention_1/attention_output/add/ReadVariableOp¢Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_1/key/add/ReadVariableOp¢7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_1/query/add/ReadVariableOp¢9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_1/value/add/ReadVariableOp¢9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp¢+sequential_1/dense_2/BiasAdd/ReadVariableOp¢-sequential_1/dense_2/Tensordot/ReadVariableOp¢+sequential_1/dense_3/BiasAdd/ReadVariableOp¢-sequential_1/dense_3/Tensordot/ReadVariableOpý
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_1/query/einsum/EinsumEinsuminputsAmulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/EinsumÛ
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/query/add/ReadVariableOpõ
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_1/query/add÷
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_1/key/einsum/EinsumEinsuminputs?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/EinsumÕ
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_1/key/add/ReadVariableOpí
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
multi_head_attention_1/key/addý
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_1/value/einsum/EinsumEinsuminputsAmulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/EinsumÛ
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/value/add/ReadVariableOpõ
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_1/value/add
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_1/Mul/yÆ
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_1/Mulü
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/EinsumÄ
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2(
&multi_head_attention_1/softmax/SoftmaxÊ
'multi_head_attention_1/dropout/IdentityIdentity0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2)
'multi_head_attention_1/dropout/Identity
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/Identity:output:0$multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/Einsum
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/Einsumø
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOp
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+multi_head_attention_1/attention_output/add
dropout_2/IdentityIdentity/multi_head_attention_1/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_2/Identityn
addAddV2inputsdropout_2/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¶
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indicesß
"layer_normalization_2/moments/meanMeanadd:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2$
"layer_normalization_2/moments/meanË
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2,
*layer_normalization_2/moments/StopGradientë
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/layer_normalization_2/moments/SquaredDifference¾
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indices
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2(
&layer_normalization_2/moments/variance
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752'
%layer_normalization_2/batchnorm/add/yê
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#layer_normalization_2/batchnorm/add¶
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2'
%layer_normalization_2/batchnorm/Rsqrtà
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOpî
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_2/batchnorm/mul½
%layer_normalization_2/batchnorm/mul_1Muladd:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_2/batchnorm/mul_1á
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_2/batchnorm/mul_2Ô
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_2/batchnorm/ReadVariableOpê
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_2/batchnorm/subá
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_2/batchnorm/add_1Õ
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02/
-sequential_1/dense_2/Tensordot/ReadVariableOp
#sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_2/Tensordot/axes
#sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_2/Tensordot/free¥
$sequential_1/dense_2/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/Shape
,sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/GatherV2/axisº
'sequential_1/dense_2/Tensordot/GatherV2GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/free:output:05sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/GatherV2¢
.sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_2/Tensordot/GatherV2_1/axisÀ
)sequential_1/dense_2/Tensordot/GatherV2_1GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/axes:output:07sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_2/Tensordot/GatherV2_1
$sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_2/Tensordot/ConstÔ
#sequential_1/dense_2/Tensordot/ProdProd0sequential_1/dense_2/Tensordot/GatherV2:output:0-sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_2/Tensordot/Prod
&sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_1Ü
%sequential_1/dense_2/Tensordot/Prod_1Prod2sequential_1/dense_2/Tensordot/GatherV2_1:output:0/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_2/Tensordot/Prod_1
*sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_2/Tensordot/concat/axis
%sequential_1/dense_2/Tensordot/concatConcatV2,sequential_1/dense_2/Tensordot/free:output:0,sequential_1/dense_2/Tensordot/axes:output:03sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_2/Tensordot/concatà
$sequential_1/dense_2/Tensordot/stackPack,sequential_1/dense_2/Tensordot/Prod:output:0.sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/stackò
(sequential_1/dense_2/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0.sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2*
(sequential_1/dense_2/Tensordot/transposeó
&sequential_1/dense_2/Tensordot/ReshapeReshape,sequential_1/dense_2/Tensordot/transpose:y:0-sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&sequential_1/dense_2/Tensordot/Reshapeò
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%sequential_1/dense_2/Tensordot/MatMul
&sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2(
&sequential_1/dense_2/Tensordot/Const_2
,sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/concat_1/axis¦
'sequential_1/dense_2/Tensordot/concat_1ConcatV20sequential_1/dense_2/Tensordot/GatherV2:output:0/sequential_1/dense_2/Tensordot/Const_2:output:05sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/concat_1ä
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:00sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2 
sequential_1/dense_2/TensordotË
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOpÛ
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_1/dense_2/BiasAdd
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_1/dense_2/ReluÕ
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02/
-sequential_1/dense_3/Tensordot/ReadVariableOp
#sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_3/Tensordot/axes
#sequential_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_3/Tensordot/free£
$sequential_1/dense_3/Tensordot/ShapeShape'sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/Shape
,sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/GatherV2/axisº
'sequential_1/dense_3/Tensordot/GatherV2GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/free:output:05sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/GatherV2¢
.sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_3/Tensordot/GatherV2_1/axisÀ
)sequential_1/dense_3/Tensordot/GatherV2_1GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/axes:output:07sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_3/Tensordot/GatherV2_1
$sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_3/Tensordot/ConstÔ
#sequential_1/dense_3/Tensordot/ProdProd0sequential_1/dense_3/Tensordot/GatherV2:output:0-sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_3/Tensordot/Prod
&sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_1Ü
%sequential_1/dense_3/Tensordot/Prod_1Prod2sequential_1/dense_3/Tensordot/GatherV2_1:output:0/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_3/Tensordot/Prod_1
*sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_3/Tensordot/concat/axis
%sequential_1/dense_3/Tensordot/concatConcatV2,sequential_1/dense_3/Tensordot/free:output:0,sequential_1/dense_3/Tensordot/axes:output:03sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_3/Tensordot/concatà
$sequential_1/dense_3/Tensordot/stackPack,sequential_1/dense_3/Tensordot/Prod:output:0.sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/stackð
(sequential_1/dense_3/Tensordot/transpose	Transpose'sequential_1/dense_2/Relu:activations:0.sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2*
(sequential_1/dense_3/Tensordot/transposeó
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_3/Tensordot/transpose:y:0-sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&sequential_1/dense_3/Tensordot/Reshapeò
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential_1/dense_3/Tensordot/MatMul
&sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_2
,sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/concat_1/axis¦
'sequential_1/dense_3/Tensordot/concat_1ConcatV20sequential_1/dense_3/Tensordot/GatherV2:output:0/sequential_1/dense_3/Tensordot/Const_2:output:05sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/concat_1ä
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:00sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
sequential_1/dense_3/TensordotË
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOpÛ
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
sequential_1/dense_3/BiasAdd
dropout_3/IdentityIdentity%sequential_1/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_3/Identity
add_1AddV2)layer_normalization_2/batchnorm/add_1:z:0dropout_3/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¶
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indicesá
"layer_normalization_3/moments/meanMean	add_1:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2$
"layer_normalization_3/moments/meanË
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2,
*layer_normalization_3/moments/StopGradientí
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/layer_normalization_3/moments/SquaredDifference¾
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indices
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2(
&layer_normalization_3/moments/variance
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752'
%layer_normalization_3/batchnorm/add/yê
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#layer_normalization_3/batchnorm/add¶
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2'
%layer_normalization_3/batchnorm/Rsqrtà
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOpî
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_3/batchnorm/mul¿
%layer_normalization_3/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_3/batchnorm/mul_1á
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_3/batchnorm/mul_2Ô
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_3/batchnorm/ReadVariableOpê
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_3/batchnorm/subá
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_3/batchnorm/add_1Ó
IdentityIdentity)layer_normalization_3/batchnorm/add_1:z:0/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2
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
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
° 
â
C__inference_dense_2_layer_call_and_return_conditional_losses_142490

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: @*
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ# ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
	
Ü
C__inference_dense_6_layer_call_and_return_conditional_losses_142310

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
è

Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_141795

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs


O__inference_batch_normalization_layer_call_and_return_conditional_losses_141549

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
æ

O__inference_batch_normalization_layer_call_and_return_conditional_losses_139479

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ÙH
§
H__inference_sequential_1_layer_call_and_return_conditional_losses_142376

inputs-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource-
)dense_3_tensordot_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity¢dense_2/BiasAdd/ReadVariableOp¢ dense_2/Tensordot/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢ dense_3/Tensordot/ReadVariableOp®
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes
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
dense_2/Tensordot/Shape
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axisù
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axisÿ
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
dense_2/Tensordot/Const 
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1¨
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axisØ
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat¬
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack¨
dense_2/Tensordot/transpose	Transposeinputs!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_2/Tensordot/transpose¿
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_2/Tensordot/Reshape¾
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_2/Tensordot/MatMul
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_2/Tensordot/Const_2
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axiså
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1°
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_2/Tensordot¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp§
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_2/BiasAddt
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_2/Relu®
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02"
 dense_3/Tensordot/ReadVariableOpz
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/axes
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
dense_3/Tensordot/Shape
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/GatherV2/axisù
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axisÿ
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
dense_3/Tensordot/Const 
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1¨
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axisØ
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat¬
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stack¼
dense_3/Tensordot/transpose	Transposedense_2/Relu:activations:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_3/Tensordot/transpose¿
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_3/Tensordot/Reshape¾
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_3/Tensordot/MatMul
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_2
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axiså
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1°
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_3/Tensordot¤
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOp§
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_3/BiasAddø
IdentityIdentitydense_3/BiasAdd:output:0^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ÁZ
Å
A__inference_model_layer_call_and_return_conditional_losses_140176
input_1
input_2'
#token_and_position_embedding_139352'
#token_and_position_embedding_139354
conv1d_139384
conv1d_139386
conv1d_1_139417
conv1d_1_139419
batch_normalization_139506
batch_normalization_139508
batch_normalization_139510
batch_normalization_139512 
batch_normalization_1_139597 
batch_normalization_1_139599 
batch_normalization_1_139601 
batch_normalization_1_139603
transformer_block_1_139972
transformer_block_1_139974
transformer_block_1_139976
transformer_block_1_139978
transformer_block_1_139980
transformer_block_1_139982
transformer_block_1_139984
transformer_block_1_139986
transformer_block_1_139988
transformer_block_1_139990
transformer_block_1_139992
transformer_block_1_139994
transformer_block_1_139996
transformer_block_1_139998
transformer_block_1_140000
transformer_block_1_140002
dense_4_140057
dense_4_140059
dense_5_140114
dense_5_140116
dense_6_140170
dense_6_140172
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢conv1d/StatefulPartitionedCall¢ conv1d_1/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall¢4token_and_position_embedding/StatefulPartitionedCall¢+transformer_block_1/StatefulPartitionedCall
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1#token_and_position_embedding_139352#token_and_position_embedding_139354*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_13934126
4token_and_position_embedding/StatefulPartitionedCallÉ
conv1d/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0conv1d_139384conv1d_139386*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_1393732 
conv1d/StatefulPartitionedCall
!average_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_1388292#
!average_pooling1d/PartitionedCallÀ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0conv1d_1_139417conv1d_1_139419*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_1394062"
 conv1d_1/StatefulPartitionedCall³
#average_pooling1d_2/PartitionedCallPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_1388592%
#average_pooling1d_2/PartitionedCall
#average_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_1388442%
#average_pooling1d_1/PartitionedCall²
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_1/PartitionedCall:output:0batch_normalization_139506batch_normalization_139508batch_normalization_139510batch_normalization_139512*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1394592-
+batch_normalization/StatefulPartitionedCallÀ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_2/PartitionedCall:output:0batch_normalization_1_139597batch_normalization_1_139599batch_normalization_1_139601batch_normalization_1_139603*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1395502/
-batch_normalization_1/StatefulPartitionedCall³
add/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:06batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_1396122
add/PartitionedCall
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0transformer_block_1_139972transformer_block_1_139974transformer_block_1_139976transformer_block_1_139978transformer_block_1_139980transformer_block_1_139982transformer_block_1_139984transformer_block_1_139986transformer_block_1_139988transformer_block_1_139990transformer_block_1_139992transformer_block_1_139994transformer_block_1_139996transformer_block_1_139998transformer_block_1_140000transformer_block_1_140002*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_1_layer_call_and_return_conditional_losses_1397692-
+transformer_block_1/StatefulPartitionedCall
flatten/PartitionedCallPartitionedCall4transformer_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1400112
flatten/PartitionedCall
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1400262
concatenate/PartitionedCall°
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_140057dense_4_140059*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1400462!
dense_4/StatefulPartitionedCall
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1400742#
!dropout_4/StatefulPartitionedCall¶
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_5_140114dense_5_140116*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1401032!
dense_5/StatefulPartitionedCall¸
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1401312#
!dropout_5/StatefulPartitionedCall¶
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_6_140170dense_6_140172*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1401592!
dense_6/StatefulPartitionedCall±
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
¸
 
-__inference_sequential_1_layer_call_fn_142459

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_1393012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
Ñ
û
H__inference_sequential_1_layer_call_and_return_conditional_losses_139301

inputs
dense_2_139290
dense_2_139292
dense_3_139295
dense_3_139297
identity¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_139290dense_2_139292*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1391802!
dense_2/StatefulPartitionedCall¸
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_139295dense_3_139297*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1392262!
dense_3/StatefulPartitionedCallÄ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
¿W
ý
A__inference_model_layer_call_and_return_conditional_losses_140540

inputs
inputs_1'
#token_and_position_embedding_140450'
#token_and_position_embedding_140452
conv1d_140455
conv1d_140457
conv1d_1_140461
conv1d_1_140463
batch_normalization_140468
batch_normalization_140470
batch_normalization_140472
batch_normalization_140474 
batch_normalization_1_140477 
batch_normalization_1_140479 
batch_normalization_1_140481 
batch_normalization_1_140483
transformer_block_1_140487
transformer_block_1_140489
transformer_block_1_140491
transformer_block_1_140493
transformer_block_1_140495
transformer_block_1_140497
transformer_block_1_140499
transformer_block_1_140501
transformer_block_1_140503
transformer_block_1_140505
transformer_block_1_140507
transformer_block_1_140509
transformer_block_1_140511
transformer_block_1_140513
transformer_block_1_140515
transformer_block_1_140517
dense_4_140522
dense_4_140524
dense_5_140528
dense_5_140530
dense_6_140534
dense_6_140536
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢conv1d/StatefulPartitionedCall¢ conv1d_1/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢4token_and_position_embedding/StatefulPartitionedCall¢+transformer_block_1/StatefulPartitionedCall
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs#token_and_position_embedding_140450#token_and_position_embedding_140452*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_13934126
4token_and_position_embedding/StatefulPartitionedCallÉ
conv1d/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0conv1d_140455conv1d_140457*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_1393732 
conv1d/StatefulPartitionedCall
!average_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_1388292#
!average_pooling1d/PartitionedCallÀ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0conv1d_1_140461conv1d_1_140463*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_1394062"
 conv1d_1/StatefulPartitionedCall³
#average_pooling1d_2/PartitionedCallPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_1388592%
#average_pooling1d_2/PartitionedCall
#average_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_1388442%
#average_pooling1d_1/PartitionedCall´
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_1/PartitionedCall:output:0batch_normalization_140468batch_normalization_140470batch_normalization_140472batch_normalization_140474*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1394792-
+batch_normalization/StatefulPartitionedCallÂ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_2/PartitionedCall:output:0batch_normalization_1_140477batch_normalization_1_140479batch_normalization_1_140481batch_normalization_1_140483*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1395702/
-batch_normalization_1/StatefulPartitionedCall³
add/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:06batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_1396122
add/PartitionedCall
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0transformer_block_1_140487transformer_block_1_140489transformer_block_1_140491transformer_block_1_140493transformer_block_1_140495transformer_block_1_140497transformer_block_1_140499transformer_block_1_140501transformer_block_1_140503transformer_block_1_140505transformer_block_1_140507transformer_block_1_140509transformer_block_1_140511transformer_block_1_140513transformer_block_1_140515transformer_block_1_140517*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_1_layer_call_and_return_conditional_losses_1398962-
+transformer_block_1/StatefulPartitionedCall
flatten/PartitionedCallPartitionedCall4transformer_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1400112
flatten/PartitionedCall
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1400262
concatenate/PartitionedCall°
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_140522dense_4_140524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1400462!
dense_4/StatefulPartitionedCallü
dropout_4/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1400792
dropout_4/PartitionedCall®
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_5_140528dense_5_140530*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1401032!
dense_5/StatefulPartitionedCallü
dropout_5/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1401362
dropout_5/PartitionedCall®
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_6_140534dense_6_140536*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1401592!
dense_6/StatefulPartitionedCallé
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
¢
&__inference_model_layer_call_fn_141332
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

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity¢StatefulPartitionedCallÐ
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1403682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
æ

H__inference_sequential_1_layer_call_and_return_conditional_losses_139257
dense_2_input
dense_2_139246
dense_2_139248
dense_3_139251
dense_3_139253
identity¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall
dense_2/StatefulPartitionedCallStatefulPartitionedCalldense_2_inputdense_2_139246dense_2_139248*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1391802!
dense_2/StatefulPartitionedCall¸
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_139251dense_3_139253*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1392262!
dense_3/StatefulPartitionedCallÄ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
'
_user_specified_namedense_2_input
º0
Æ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_141611

inputs
assignmovingavg_141586
assignmovingavg_1_141592)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/141586*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_141586*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/141586*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/141586*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_141586AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/141586*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/141592*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_141592*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/141592*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/141592*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_141592AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/141592*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
Ð
â
C__inference_dense_3_layer_call_and_return_conditional_losses_139226

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@ *
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ#@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@
 
_user_specified_nameinputs
ê
§
4__inference_batch_normalization_layer_call_fn_141575

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1389942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¸
q
G__inference_concatenate_layer_call_and_return_conditional_losses_140026

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿà:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÁW
ý
A__inference_model_layer_call_and_return_conditional_losses_140270
input_1
input_2'
#token_and_position_embedding_140180'
#token_and_position_embedding_140182
conv1d_140185
conv1d_140187
conv1d_1_140191
conv1d_1_140193
batch_normalization_140198
batch_normalization_140200
batch_normalization_140202
batch_normalization_140204 
batch_normalization_1_140207 
batch_normalization_1_140209 
batch_normalization_1_140211 
batch_normalization_1_140213
transformer_block_1_140217
transformer_block_1_140219
transformer_block_1_140221
transformer_block_1_140223
transformer_block_1_140225
transformer_block_1_140227
transformer_block_1_140229
transformer_block_1_140231
transformer_block_1_140233
transformer_block_1_140235
transformer_block_1_140237
transformer_block_1_140239
transformer_block_1_140241
transformer_block_1_140243
transformer_block_1_140245
transformer_block_1_140247
dense_4_140252
dense_4_140254
dense_5_140258
dense_5_140260
dense_6_140264
dense_6_140266
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢conv1d/StatefulPartitionedCall¢ conv1d_1/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢4token_and_position_embedding/StatefulPartitionedCall¢+transformer_block_1/StatefulPartitionedCall
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1#token_and_position_embedding_140180#token_and_position_embedding_140182*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_13934126
4token_and_position_embedding/StatefulPartitionedCallÉ
conv1d/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0conv1d_140185conv1d_140187*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_1393732 
conv1d/StatefulPartitionedCall
!average_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_1388292#
!average_pooling1d/PartitionedCallÀ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0conv1d_1_140191conv1d_1_140193*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_1394062"
 conv1d_1/StatefulPartitionedCall³
#average_pooling1d_2/PartitionedCallPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_1388592%
#average_pooling1d_2/PartitionedCall
#average_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_1388442%
#average_pooling1d_1/PartitionedCall´
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_1/PartitionedCall:output:0batch_normalization_140198batch_normalization_140200batch_normalization_140202batch_normalization_140204*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1394792-
+batch_normalization/StatefulPartitionedCallÂ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_2/PartitionedCall:output:0batch_normalization_1_140207batch_normalization_1_140209batch_normalization_1_140211batch_normalization_1_140213*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1395702/
-batch_normalization_1/StatefulPartitionedCall³
add/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:06batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_1396122
add/PartitionedCall
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0transformer_block_1_140217transformer_block_1_140219transformer_block_1_140221transformer_block_1_140223transformer_block_1_140225transformer_block_1_140227transformer_block_1_140229transformer_block_1_140231transformer_block_1_140233transformer_block_1_140235transformer_block_1_140237transformer_block_1_140239transformer_block_1_140241transformer_block_1_140243transformer_block_1_140245transformer_block_1_140247*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_1_layer_call_and_return_conditional_losses_1398962-
+transformer_block_1/StatefulPartitionedCall
flatten/PartitionedCallPartitionedCall4transformer_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1400112
flatten/PartitionedCall
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1400262
concatenate/PartitionedCall°
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_140252dense_4_140254*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1400462!
dense_4/StatefulPartitionedCallü
dropout_4/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1400792
dropout_4/PartitionedCall®
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_5_140258dense_5_140260*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1401032!
dense_5/StatefulPartitionedCallü
dropout_5/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1401362
dropout_5/PartitionedCall®
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_6_140264dense_6_140266*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1401592!
dense_6/StatefulPartitionedCallé
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
ó
~
)__inference_conv1d_1_layer_call_fn_141493

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_1394062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÞ ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 
 
_user_specified_nameinputs
Ð
â
C__inference_dense_3_layer_call_and_return_conditional_losses_142529

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@ *
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ#@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@
 
_user_specified_nameinputs
Ñ
û
H__inference_sequential_1_layer_call_and_return_conditional_losses_139274

inputs
dense_2_139263
dense_2_139265
dense_3_139268
dense_3_139270
identity¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_139263dense_2_139265*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1391802!
dense_2/StatefulPartitionedCall¸
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_139268dense_3_139270*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1392262!
dense_3/StatefulPartitionedCallÄ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
Ê
þ
X__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_139341
x'
#embedding_1_embedding_lookup_139328%
!embedding_embedding_lookup_139334
identity¢embedding/embedding_lookup¢embedding_1/embedding_lookup?
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
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice/stack_2â
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
range/delta
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
range¯
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_139328range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_1/embedding_lookup/139328*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02
embedding_1/embedding_lookup
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/139328*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%embedding_1/embedding_lookup/IdentityÀ
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'embedding_1/embedding_lookup/Identity_1m
embedding/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2
embedding/Cast°
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_139334embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/139334*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype02
embedding/embedding_lookup
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/139334*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2%
#embedding/embedding_lookup/Identity¿
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2'
%embedding/embedding_lookup/Identity_1¬
addAddV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
add
IdentityIdentityadd:z:0^embedding/embedding_lookup^embedding_1/embedding_lookup*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿR::28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:K G
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR

_user_specified_namex
æ

H__inference_sequential_1_layer_call_and_return_conditional_losses_139243
dense_2_input
dense_2_139191
dense_2_139193
dense_3_139237
dense_3_139239
identity¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall
dense_2/StatefulPartitionedCallStatefulPartitionedCalldense_2_inputdense_2_139191dense_2_139193*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1391802!
dense_2/StatefulPartitionedCall¸
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_139237dense_3_139239*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1392262!
dense_3/StatefulPartitionedCallÄ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
'
_user_specified_namedense_2_input
õ
k
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_138844

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

ExpandDimsº
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize

*
paddingVALID*
strides

2	
AvgPool
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó0
È
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_139101

inputs
assignmovingavg_139076
assignmovingavg_1_139082)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/139076*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_139076*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/139076*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/139076*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_139076AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/139076*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/139082*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_139082*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/139082*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/139082*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_139082AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/139082*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ð	
Ü
C__inference_dense_4_layer_call_and_return_conditional_losses_142217

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿè::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
 
_user_specified_nameinputs
µ
i
?__inference_add_layer_call_and_return_conditional_losses_139612

inputs
inputs_1
identity[
addAddV2inputsinputs_1*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ# :ÿÿÿÿÿÿÿÿÿ# :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs:SO
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
î
©
6__inference_batch_normalization_1_layer_call_fn_141739

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1391342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


O__inference_batch_normalization_layer_call_and_return_conditional_losses_138994

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Í
§
-__inference_sequential_1_layer_call_fn_139285
dense_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCalldense_2_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_1392742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
'
_user_specified_namedense_2_input
ß
}
(__inference_dense_4_layer_call_fn_142226

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1400462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿè::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
 
_user_specified_nameinputs
Ê
©
6__inference_batch_normalization_1_layer_call_fn_141821

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1395702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ì
©
6__inference_batch_normalization_1_layer_call_fn_141726

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1391012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ì
 
&__inference_model_layer_call_fn_140443
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

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity¢StatefulPartitionedCallÎ
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1403682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2


Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_139134

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1è
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

d
E__inference_dropout_5_layer_call_and_return_conditional_losses_140131

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¼0
È
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_139550

inputs
assignmovingavg_139525
assignmovingavg_1_139531)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient¨
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/139525*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_139525*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/139525*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/139525*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_139525AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/139525*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/139531*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_139531*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/139531*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/139531*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_139531AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/139531*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/add_1·
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

N
2__inference_average_pooling1d_layer_call_fn_138835

inputs
identityä
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_1388292
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
§
4__inference_batch_normalization_layer_call_fn_141644

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1394592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
ìÕ
$
A__inference_model_layer_call_and_return_conditional_losses_141254
inputs_0
inputs_1D
@token_and_position_embedding_embedding_1_embedding_lookup_141023B
>token_and_position_embedding_embedding_embedding_lookup_1410296
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource=
9batch_normalization_batchnorm_mul_readvariableop_resource;
7batch_normalization_batchnorm_readvariableop_1_resource;
7batch_normalization_batchnorm_readvariableop_2_resource;
7batch_normalization_1_batchnorm_readvariableop_resource?
;batch_normalization_1_batchnorm_mul_readvariableop_resource=
9batch_normalization_1_batchnorm_readvariableop_1_resource=
9batch_normalization_1_batchnorm_readvariableop_2_resourceZ
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
identity¢,batch_normalization/batchnorm/ReadVariableOp¢.batch_normalization/batchnorm/ReadVariableOp_1¢.batch_normalization/batchnorm/ReadVariableOp_2¢0batch_normalization/batchnorm/mul/ReadVariableOp¢.batch_normalization_1/batchnorm/ReadVariableOp¢0batch_normalization_1/batchnorm/ReadVariableOp_1¢0batch_normalization_1/batchnorm/ReadVariableOp_2¢2batch_normalization_1/batchnorm/mul/ReadVariableOp¢conv1d/BiasAdd/ReadVariableOp¢)conv1d/conv1d/ExpandDims_1/ReadVariableOp¢conv1d_1/BiasAdd/ReadVariableOp¢+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢7token_and_position_embedding/embedding/embedding_lookup¢9token_and_position_embedding/embedding_1/embedding_lookup¢Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp¢Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp¢Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp¢Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp¢Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp¢Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp¢Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp¢Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp¢Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp¢Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp¢Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp¢Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp¢?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp¢Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp¢?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp¢Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp
"token_and_position_embedding/ShapeShapeinputs_0*
T0*
_output_shapes
:2$
"token_and_position_embedding/Shape·
0token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ22
0token_and_position_embedding/strided_slice/stack²
2token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 24
2token_and_position_embedding/strided_slice/stack_1²
2token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2token_and_position_embedding/strided_slice/stack_2
*token_and_position_embedding/strided_sliceStridedSlice+token_and_position_embedding/Shape:output:09token_and_position_embedding/strided_slice/stack:output:0;token_and_position_embedding/strided_slice/stack_1:output:0;token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*token_and_position_embedding/strided_slice
(token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2*
(token_and_position_embedding/range/start
(token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2*
(token_and_position_embedding/range/delta
"token_and_position_embedding/rangeRange1token_and_position_embedding/range/start:output:03token_and_position_embedding/strided_slice:output:01token_and_position_embedding/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"token_and_position_embedding/rangeÀ
9token_and_position_embedding/embedding_1/embedding_lookupResourceGather@token_and_position_embedding_embedding_1_embedding_lookup_141023+token_and_position_embedding/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*S
_classI
GEloc:@token_and_position_embedding/embedding_1/embedding_lookup/141023*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype02;
9token_and_position_embedding/embedding_1/embedding_lookup
Btoken_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityBtoken_and_position_embedding/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*S
_classI
GEloc:@token_and_position_embedding/embedding_1/embedding_lookup/141023*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2D
Btoken_and_position_embedding/embedding_1/embedding_lookup/Identity
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityKtoken_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2F
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1®
+token_and_position_embedding/embedding/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR2-
+token_and_position_embedding/embedding/CastÁ
7token_and_position_embedding/embedding/embedding_lookupResourceGather>token_and_position_embedding_embedding_embedding_lookup_141029/token_and_position_embedding/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@token_and_position_embedding/embedding/embedding_lookup/141029*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
dtype029
7token_and_position_embedding/embedding/embedding_lookup
@token_and_position_embedding/embedding/embedding_lookup/IdentityIdentity@token_and_position_embedding/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@token_and_position_embedding/embedding/embedding_lookup/141029*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2B
@token_and_position_embedding/embedding/embedding_lookup/Identity
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityItoken_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2D
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1 
 token_and_position_embedding/addAddV2Ktoken_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Mtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2"
 token_and_position_embedding/add
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/conv1d/ExpandDims/dimÊ
conv1d/conv1d/ExpandDims
ExpandDims$token_and_position_embedding/add:z:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d/conv1d/ExpandDimsÍ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimÓ
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/conv1d/ExpandDims_1Ó
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
paddingSAME*
strides
2
conv1d/conv1d¨
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/conv1d/Squeeze¡
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1d/BiasAdd/ReadVariableOp©
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d/BiasAddr
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d/Relu
 average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 average_pooling1d/ExpandDims/dimË
average_pooling1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0)average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
average_pooling1d/ExpandDimsß
average_pooling1d/AvgPoolAvgPool%average_pooling1d/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
ksize
*
paddingVALID*
strides
2
average_pooling1d/AvgPool³
average_pooling1d/SqueezeSqueeze"average_pooling1d/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims
2
average_pooling1d/Squeeze
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2 
conv1d_1/conv1d/ExpandDims/dimÎ
conv1d_1/conv1d/ExpandDims
ExpandDims"average_pooling1d/Squeeze:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_1/conv1d/ExpandDimsÓ
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dimÛ
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d_1/conv1d/ExpandDims_1Û
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
conv1d_1/conv1d®
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d_1/conv1d/Squeeze§
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_1/BiasAdd/ReadVariableOp±
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_1/BiasAddx
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d_1/Relu
"average_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_2/ExpandDims/dimÜ
average_pooling1d_2/ExpandDims
ExpandDims$token_and_position_embedding/add:z:0+average_pooling1d_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2 
average_pooling1d_2/ExpandDimsæ
average_pooling1d_2/AvgPoolAvgPool'average_pooling1d_2/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize	
¬*
paddingVALID*
strides	
¬2
average_pooling1d_2/AvgPool¸
average_pooling1d_2/SqueezeSqueeze$average_pooling1d_2/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_2/Squeeze
"average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_1/ExpandDims/dimÓ
average_pooling1d_1/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0+average_pooling1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2 
average_pooling1d_1/ExpandDimsä
average_pooling1d_1/AvgPoolAvgPool'average_pooling1d_1/ExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
ksize

*
paddingVALID*
strides

2
average_pooling1d_1/AvgPool¸
average_pooling1d_1/SqueezeSqueeze$average_pooling1d_1/AvgPool:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
squeeze_dims
2
average_pooling1d_1/SqueezeÎ
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02.
,batch_normalization/batchnorm/ReadVariableOp
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2%
#batch_normalization/batchnorm/add/yØ
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/add
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/RsqrtÚ
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOpÕ
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/mulÔ
#batch_normalization/batchnorm/mul_1Mul$average_pooling1d_1/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#batch_normalization/batchnorm/mul_1Ô
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1Õ
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
: 2%
#batch_normalization/batchnorm/mul_2Ô
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2Ó
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2#
!batch_normalization/batchnorm/subÙ
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#batch_normalization/batchnorm/add_1Ô
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_1/batchnorm/add/yà
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/add¥
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_1/batchnorm/Rsqrtà
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOpÝ
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/mulÚ
%batch_normalization_1/batchnorm/mul_1Mul$average_pooling1d_2/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_1/batchnorm/mul_1Ú
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1Ý
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_1/batchnorm/mul_2Ú
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2Û
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/subá
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%batch_normalization_1/batchnorm/add_1¥
add/addAddV2'batch_normalization/batchnorm/add_1:z:0)batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2	
add/add¹
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_1_multi_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpÎ
>transformer_block_1/multi_head_attention_1/query/einsum/EinsumEinsumadd/add:z:0Utransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_1/multi_head_attention_1/query/einsum/Einsum
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOpReadVariableOpLtransformer_block_1_multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOpÅ
4transformer_block_1/multi_head_attention_1/query/addAddV2Gtransformer_block_1/multi_head_attention_1/query/einsum/Einsum:output:0Ktransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_1/multi_head_attention_1/query/add³
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_1_multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpÈ
<transformer_block_1/multi_head_attention_1/key/einsum/EinsumEinsumadd/add:z:0Stransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2>
<transformer_block_1/multi_head_attention_1/key/einsum/Einsum
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOpReadVariableOpJtransformer_block_1_multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp½
2transformer_block_1/multi_head_attention_1/key/addAddV2Etransformer_block_1/multi_head_attention_1/key/einsum/Einsum:output:0Itransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 24
2transformer_block_1/multi_head_attention_1/key/add¹
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_1_multi_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpÎ
>transformer_block_1/multi_head_attention_1/value/einsum/EinsumEinsumadd/add:z:0Utransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2@
>transformer_block_1/multi_head_attention_1/value/einsum/Einsum
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOpReadVariableOpLtransformer_block_1_multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOpÅ
4transformer_block_1/multi_head_attention_1/value/addAddV2Gtransformer_block_1/multi_head_attention_1/value/einsum/Einsum:output:0Ktransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 26
4transformer_block_1/multi_head_attention_1/value/add©
0transformer_block_1/multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>22
0transformer_block_1/multi_head_attention_1/Mul/y
.transformer_block_1/multi_head_attention_1/MulMul8transformer_block_1/multi_head_attention_1/query/add:z:09transformer_block_1/multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 20
.transformer_block_1/multi_head_attention_1/MulÌ
8transformer_block_1/multi_head_attention_1/einsum/EinsumEinsum6transformer_block_1/multi_head_attention_1/key/add:z:02transformer_block_1/multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2:
8transformer_block_1/multi_head_attention_1/einsum/Einsum
:transformer_block_1/multi_head_attention_1/softmax/SoftmaxSoftmaxAtransformer_block_1/multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2<
:transformer_block_1/multi_head_attention_1/softmax/Softmax
;transformer_block_1/multi_head_attention_1/dropout/IdentityIdentityDtransformer_block_1/multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2=
;transformer_block_1/multi_head_attention_1/dropout/Identityä
:transformer_block_1/multi_head_attention_1/einsum_1/EinsumEinsumDtransformer_block_1/multi_head_attention_1/dropout/Identity:output:08transformer_block_1/multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2<
:transformer_block_1/multi_head_attention_1/einsum_1/EinsumÚ
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_1_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp£
Itransformer_block_1/multi_head_attention_1/attention_output/einsum/EinsumEinsumCtransformer_block_1/multi_head_attention_1/einsum_1/Einsum:output:0`transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe2K
Itransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum´
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_1_multi_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpí
?transformer_block_1/multi_head_attention_1/attention_output/addAddV2Rtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum:output:0Vtransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2A
?transformer_block_1/multi_head_attention_1/attention_output/add×
&transformer_block_1/dropout_2/IdentityIdentityCtransformer_block_1/multi_head_attention_1/attention_output/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&transformer_block_1/dropout_2/Identity¯
transformer_block_1/addAddV2add/add:z:0/transformer_block_1/dropout_2/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_1/addÞ
Htransformer_block_1/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_1/layer_normalization_2/moments/mean/reduction_indices¯
6transformer_block_1/layer_normalization_2/moments/meanMeantransformer_block_1/add:z:0Qtransformer_block_1/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(28
6transformer_block_1/layer_normalization_2/moments/mean
>transformer_block_1/layer_normalization_2/moments/StopGradientStopGradient?transformer_block_1/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2@
>transformer_block_1/layer_normalization_2/moments/StopGradient»
Ctransformer_block_1/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencetransformer_block_1/add:z:0Gtransformer_block_1/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2E
Ctransformer_block_1/layer_normalization_2/moments/SquaredDifferenceæ
Ltransformer_block_1/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_1/layer_normalization_2/moments/variance/reduction_indicesç
:transformer_block_1/layer_normalization_2/moments/varianceMeanGtransformer_block_1/layer_normalization_2/moments/SquaredDifference:z:0Utransformer_block_1/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2<
:transformer_block_1/layer_normalization_2/moments/variance»
9transformer_block_1/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752;
9transformer_block_1/layer_normalization_2/batchnorm/add/yº
7transformer_block_1/layer_normalization_2/batchnorm/addAddV2Ctransformer_block_1/layer_normalization_2/moments/variance:output:0Btransformer_block_1/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#29
7transformer_block_1/layer_normalization_2/batchnorm/addò
9transformer_block_1/layer_normalization_2/batchnorm/RsqrtRsqrt;transformer_block_1/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2;
9transformer_block_1/layer_normalization_2/batchnorm/Rsqrt
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp¾
7transformer_block_1/layer_normalization_2/batchnorm/mulMul=transformer_block_1/layer_normalization_2/batchnorm/Rsqrt:y:0Ntransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 29
7transformer_block_1/layer_normalization_2/batchnorm/mul
9transformer_block_1/layer_normalization_2/batchnorm/mul_1Multransformer_block_1/add:z:0;transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_1/layer_normalization_2/batchnorm/mul_1±
9transformer_block_1/layer_normalization_2/batchnorm/mul_2Mul?transformer_block_1/layer_normalization_2/moments/mean:output:0;transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_1/layer_normalization_2/batchnorm/mul_2
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_1_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpº
7transformer_block_1/layer_normalization_2/batchnorm/subSubJtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp:value:0=transformer_block_1/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 29
7transformer_block_1/layer_normalization_2/batchnorm/sub±
9transformer_block_1/layer_normalization_2/batchnorm/add_1AddV2=transformer_block_1/layer_normalization_2/batchnorm/mul_1:z:0;transformer_block_1/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_1/layer_normalization_2/batchnorm/add_1
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_1_sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02C
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp¼
7transformer_block_1/sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_1/sequential_1/dense_2/Tensordot/axesÃ
7transformer_block_1/sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_1/sequential_1/dense_2/Tensordot/freeá
8transformer_block_1/sequential_1/dense_2/Tensordot/ShapeShape=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_2/Tensordot/ShapeÆ
@transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axis
;transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2GatherV2Atransformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2Ê
Btransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axis¤
=transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1GatherV2Atransformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Ktransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1¾
8transformer_block_1/sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_1/sequential_1/dense_2/Tensordot/Const¤
7transformer_block_1/sequential_1/dense_2/Tensordot/ProdProdDtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Atransformer_block_1/sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_1/sequential_1/dense_2/Tensordot/ProdÂ
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_1¬
9transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1ProdFtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1:output:0Ctransformer_block_1/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1Â
>transformer_block_1/sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_1/sequential_1/dense_2/Tensordot/concat/axisý
9transformer_block_1/sequential_1/dense_2/Tensordot/concatConcatV2@transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Gtransformer_block_1/sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_1/sequential_1/dense_2/Tensordot/concat°
8transformer_block_1/sequential_1/dense_2/Tensordot/stackPack@transformer_block_1/sequential_1/dense_2/Tensordot/Prod:output:0Btransformer_block_1/sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_2/Tensordot/stackÂ
<transformer_block_1/sequential_1/dense_2/Tensordot/transpose	Transpose=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0Btransformer_block_1/sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2>
<transformer_block_1/sequential_1/dense_2/Tensordot/transposeÃ
:transformer_block_1/sequential_1/dense_2/Tensordot/ReshapeReshape@transformer_block_1/sequential_1/dense_2/Tensordot/transpose:y:0Atransformer_block_1/sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2<
:transformer_block_1/sequential_1/dense_2/Tensordot/ReshapeÂ
9transformer_block_1/sequential_1/dense_2/Tensordot/MatMulMatMulCtransformer_block_1/sequential_1/dense_2/Tensordot/Reshape:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2;
9transformer_block_1/sequential_1/dense_2/Tensordot/MatMulÂ
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2<
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_2Æ
@transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axis
;transformer_block_1/sequential_1/dense_2/Tensordot/concat_1ConcatV2Dtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Ctransformer_block_1/sequential_1/dense_2/Tensordot/Const_2:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_2/Tensordot/concat_1´
2transformer_block_1/sequential_1/dense_2/TensordotReshapeCtransformer_block_1/sequential_1/dense_2/Tensordot/MatMul:product:0Dtransformer_block_1/sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@24
2transformer_block_1/sequential_1/dense_2/Tensordot
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_1_sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02A
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp«
0transformer_block_1/sequential_1/dense_2/BiasAddBiasAdd;transformer_block_1/sequential_1/dense_2/Tensordot:output:0Gtransformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@22
0transformer_block_1/sequential_1/dense_2/BiasAdd×
-transformer_block_1/sequential_1/dense_2/ReluRelu9transformer_block_1/sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2/
-transformer_block_1/sequential_1/dense_2/Relu
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_1_sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02C
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp¼
7transformer_block_1/sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_1/sequential_1/dense_3/Tensordot/axesÃ
7transformer_block_1/sequential_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_1/sequential_1/dense_3/Tensordot/freeß
8transformer_block_1/sequential_1/dense_3/Tensordot/ShapeShape;transformer_block_1/sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_3/Tensordot/ShapeÆ
@transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axis
;transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2GatherV2Atransformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2Ê
Btransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axis¤
=transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1GatherV2Atransformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Ktransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1¾
8transformer_block_1/sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_1/sequential_1/dense_3/Tensordot/Const¤
7transformer_block_1/sequential_1/dense_3/Tensordot/ProdProdDtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Atransformer_block_1/sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_1/sequential_1/dense_3/Tensordot/ProdÂ
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_1¬
9transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1ProdFtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1:output:0Ctransformer_block_1/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1Â
>transformer_block_1/sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_1/sequential_1/dense_3/Tensordot/concat/axisý
9transformer_block_1/sequential_1/dense_3/Tensordot/concatConcatV2@transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Gtransformer_block_1/sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_1/sequential_1/dense_3/Tensordot/concat°
8transformer_block_1/sequential_1/dense_3/Tensordot/stackPack@transformer_block_1/sequential_1/dense_3/Tensordot/Prod:output:0Btransformer_block_1/sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_3/Tensordot/stackÀ
<transformer_block_1/sequential_1/dense_3/Tensordot/transpose	Transpose;transformer_block_1/sequential_1/dense_2/Relu:activations:0Btransformer_block_1/sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2>
<transformer_block_1/sequential_1/dense_3/Tensordot/transposeÃ
:transformer_block_1/sequential_1/dense_3/Tensordot/ReshapeReshape@transformer_block_1/sequential_1/dense_3/Tensordot/transpose:y:0Atransformer_block_1/sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2<
:transformer_block_1/sequential_1/dense_3/Tensordot/ReshapeÂ
9transformer_block_1/sequential_1/dense_3/Tensordot/MatMulMatMulCtransformer_block_1/sequential_1/dense_3/Tensordot/Reshape:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2;
9transformer_block_1/sequential_1/dense_3/Tensordot/MatMulÂ
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_2Æ
@transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axis
;transformer_block_1/sequential_1/dense_3/Tensordot/concat_1ConcatV2Dtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Ctransformer_block_1/sequential_1/dense_3/Tensordot/Const_2:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_3/Tensordot/concat_1´
2transformer_block_1/sequential_1/dense_3/TensordotReshapeCtransformer_block_1/sequential_1/dense_3/Tensordot/MatMul:product:0Dtransformer_block_1/sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 24
2transformer_block_1/sequential_1/dense_3/Tensordot
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_1_sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp«
0transformer_block_1/sequential_1/dense_3/BiasAddBiasAdd;transformer_block_1/sequential_1/dense_3/Tensordot:output:0Gtransformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 22
0transformer_block_1/sequential_1/dense_3/BiasAddÍ
&transformer_block_1/dropout_3/IdentityIdentity9transformer_block_1/sequential_1/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2(
&transformer_block_1/dropout_3/Identityå
transformer_block_1/add_1AddV2=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0/transformer_block_1/dropout_3/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
transformer_block_1/add_1Þ
Htransformer_block_1/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_1/layer_normalization_3/moments/mean/reduction_indices±
6transformer_block_1/layer_normalization_3/moments/meanMeantransformer_block_1/add_1:z:0Qtransformer_block_1/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(28
6transformer_block_1/layer_normalization_3/moments/mean
>transformer_block_1/layer_normalization_3/moments/StopGradientStopGradient?transformer_block_1/layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2@
>transformer_block_1/layer_normalization_3/moments/StopGradient½
Ctransformer_block_1/layer_normalization_3/moments/SquaredDifferenceSquaredDifferencetransformer_block_1/add_1:z:0Gtransformer_block_1/layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2E
Ctransformer_block_1/layer_normalization_3/moments/SquaredDifferenceæ
Ltransformer_block_1/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_1/layer_normalization_3/moments/variance/reduction_indicesç
:transformer_block_1/layer_normalization_3/moments/varianceMeanGtransformer_block_1/layer_normalization_3/moments/SquaredDifference:z:0Utransformer_block_1/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2<
:transformer_block_1/layer_normalization_3/moments/variance»
9transformer_block_1/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752;
9transformer_block_1/layer_normalization_3/batchnorm/add/yº
7transformer_block_1/layer_normalization_3/batchnorm/addAddV2Ctransformer_block_1/layer_normalization_3/moments/variance:output:0Btransformer_block_1/layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#29
7transformer_block_1/layer_normalization_3/batchnorm/addò
9transformer_block_1/layer_normalization_3/batchnorm/RsqrtRsqrt;transformer_block_1/layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2;
9transformer_block_1/layer_normalization_3/batchnorm/Rsqrt
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp¾
7transformer_block_1/layer_normalization_3/batchnorm/mulMul=transformer_block_1/layer_normalization_3/batchnorm/Rsqrt:y:0Ntransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 29
7transformer_block_1/layer_normalization_3/batchnorm/mul
9transformer_block_1/layer_normalization_3/batchnorm/mul_1Multransformer_block_1/add_1:z:0;transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_1/layer_normalization_3/batchnorm/mul_1±
9transformer_block_1/layer_normalization_3/batchnorm/mul_2Mul?transformer_block_1/layer_normalization_3/moments/mean:output:0;transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_1/layer_normalization_3/batchnorm/mul_2
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpº
7transformer_block_1/layer_normalization_3/batchnorm/subSubJtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp:value:0=transformer_block_1/layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 29
7transformer_block_1/layer_normalization_3/batchnorm/sub±
9transformer_block_1/layer_normalization_3/batchnorm/add_1AddV2=transformer_block_1/layer_normalization_3/batchnorm/mul_1:z:0;transformer_block_1/layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2;
9transformer_block_1/layer_normalization_3/batchnorm/add_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
flatten/Const·
flatten/ReshapeReshape=transformer_block_1/layer_normalization_3/batchnorm/add_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
flatten/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis¶
concatenate/concatConcatV2flatten/Reshape:output:0inputs_1 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
concatenate/concat¦
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02
dense_4/MatMul/ReadVariableOp 
dense_4/MatMulMatMulconcatenate/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_4/MatMul¤
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_4/BiasAdd/ReadVariableOp¡
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_4/Relu
dropout_4/IdentityIdentitydense_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_4/Identity¥
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_5/MatMul/ReadVariableOp 
dense_5/MatMulMatMuldropout_4/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_5/MatMul¤
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_5/BiasAdd/ReadVariableOp¡
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_5/Relu
dropout_5/IdentityIdentitydense_5/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_5/Identity¥
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_6/MatMul/ReadVariableOp 
dense_6/MatMulMatMuldropout_5/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/MatMul¤
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp¡
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/BiasAddã
IdentityIdentitydense_6/BiasAdd:output:0-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp8^token_and_position_embedding/embedding/embedding_lookup:^token_and_position_embedding/embedding_1/embedding_lookupC^transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpG^transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpC^transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpG^transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpO^transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpY^transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_1/multi_head_attention_1/key/add/ReadVariableOpL^transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpD^transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpN^transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpD^transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpN^transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp@^transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpB^transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp@^transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpB^transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2r
7token_and_position_embedding/embedding/embedding_lookup7token_and_position_embedding/embedding/embedding_lookup2v
9token_and_position_embedding/embedding_1/embedding_lookup9token_and_position_embedding/embedding_1/embedding_lookup2
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpBtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp2
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpFtransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp2
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpBtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp2
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpFtransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp2 
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpNtransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp2´
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOpAtransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp2
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpKtransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOpCtransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp2
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpMtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOpCtransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp2
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpMtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp2
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpAtransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp2
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp2
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpAtransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ð

à
4__inference_transformer_block_1_layer_call_fn_142145

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
identity¢StatefulPartitionedCallÁ
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
:ÿÿÿÿÿÿÿÿÿ# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_1_layer_call_and_return_conditional_losses_1397692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
í
}
(__inference_dense_2_layer_call_fn_142499

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_1391802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ# ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

÷
D__inference_conv1d_1_layer_call_and_return_conditional_losses_141484

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÞ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ 
 
_user_specified_nameinputs
®

$__inference_signature_wrapper_140701
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

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity¢StatefulPartitionedCall²
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_1388202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2

P
4__inference_average_pooling1d_2_layer_call_fn_138865

inputs
identityæ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_1388592
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯ü
Ò
O__inference_transformer_block_1_layer_call_and_return_conditional_losses_139769

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
identity¢.layer_normalization_2/batchnorm/ReadVariableOp¢2layer_normalization_2/batchnorm/mul/ReadVariableOp¢.layer_normalization_3/batchnorm/ReadVariableOp¢2layer_normalization_3/batchnorm/mul/ReadVariableOp¢:multi_head_attention_1/attention_output/add/ReadVariableOp¢Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp¢-multi_head_attention_1/key/add/ReadVariableOp¢7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp¢/multi_head_attention_1/query/add/ReadVariableOp¢9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp¢/multi_head_attention_1/value/add/ReadVariableOp¢9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp¢+sequential_1/dense_2/BiasAdd/ReadVariableOp¢-sequential_1/dense_2/Tensordot/ReadVariableOp¢+sequential_1/dense_3/BiasAdd/ReadVariableOp¢-sequential_1/dense_3/Tensordot/ReadVariableOpý
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp
*multi_head_attention_1/query/einsum/EinsumEinsuminputsAmulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/EinsumÛ
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/query/add/ReadVariableOpõ
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_1/query/add÷
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp
(multi_head_attention_1/key/einsum/EinsumEinsuminputs?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/EinsumÕ
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_1/key/add/ReadVariableOpí
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
multi_head_attention_1/key/addý
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp
*multi_head_attention_1/value/einsum/EinsumEinsuminputsAmulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/EinsumÛ
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/value/add/ReadVariableOpõ
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2"
 multi_head_attention_1/value/add
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ó5>2
multi_head_attention_1/Mul/yÆ
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
multi_head_attention_1/Mulü
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/EinsumÄ
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2(
&multi_head_attention_1/softmax/Softmax¡
,multi_head_attention_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,multi_head_attention_1/dropout/dropout/Const
*multi_head_attention_1/dropout/dropout/MulMul0multi_head_attention_1/softmax/Softmax:softmax:05multi_head_attention_1/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2,
*multi_head_attention_1/dropout/dropout/Mul¼
,multi_head_attention_1/dropout/dropout/ShapeShape0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_1/dropout/dropout/Shape
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_1/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##*
dtype02E
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform³
5multi_head_attention_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_1/dropout/dropout/GreaterEqual/yÂ
3multi_head_attention_1/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_1/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##25
3multi_head_attention_1/dropout/dropout/GreaterEqualä
+multi_head_attention_1/dropout/dropout/CastCast7multi_head_attention_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2-
+multi_head_attention_1/dropout/dropout/Castþ
,multi_head_attention_1/dropout/dropout/Mul_1Mul.multi_head_attention_1/dropout/dropout/Mul:z:0/multi_head_attention_1/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ##2.
,multi_head_attention_1/dropout/dropout/Mul_1
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/dropout/Mul_1:z:0$multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/Einsum
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpÓ
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/Einsumø
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOp
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2-
+multi_head_attention_1/attention_output/addw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_2/dropout/Const¾
dropout_2/dropout/MulMul/multi_head_attention_1/attention_output/add:z:0 dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_2/dropout/Mul
dropout_2/dropout/ShapeShape/multi_head_attention_1/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_2/dropout/ShapeÖ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_2/dropout/GreaterEqual/yê
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
dropout_2/dropout/GreaterEqual¡
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_2/dropout/Cast¦
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_2/dropout/Mul_1n
addAddV2inputsdropout_2/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add¶
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indicesß
"layer_normalization_2/moments/meanMeanadd:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2$
"layer_normalization_2/moments/meanË
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2,
*layer_normalization_2/moments/StopGradientë
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/layer_normalization_2/moments/SquaredDifference¾
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indices
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2(
&layer_normalization_2/moments/variance
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752'
%layer_normalization_2/batchnorm/add/yê
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#layer_normalization_2/batchnorm/add¶
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2'
%layer_normalization_2/batchnorm/Rsqrtà
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOpî
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_2/batchnorm/mul½
%layer_normalization_2/batchnorm/mul_1Muladd:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_2/batchnorm/mul_1á
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_2/batchnorm/mul_2Ô
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_2/batchnorm/ReadVariableOpê
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_2/batchnorm/subá
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_2/batchnorm/add_1Õ
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02/
-sequential_1/dense_2/Tensordot/ReadVariableOp
#sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_2/Tensordot/axes
#sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_2/Tensordot/free¥
$sequential_1/dense_2/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/Shape
,sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/GatherV2/axisº
'sequential_1/dense_2/Tensordot/GatherV2GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/free:output:05sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/GatherV2¢
.sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_2/Tensordot/GatherV2_1/axisÀ
)sequential_1/dense_2/Tensordot/GatherV2_1GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/axes:output:07sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_2/Tensordot/GatherV2_1
$sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_2/Tensordot/ConstÔ
#sequential_1/dense_2/Tensordot/ProdProd0sequential_1/dense_2/Tensordot/GatherV2:output:0-sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_2/Tensordot/Prod
&sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_1Ü
%sequential_1/dense_2/Tensordot/Prod_1Prod2sequential_1/dense_2/Tensordot/GatherV2_1:output:0/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_2/Tensordot/Prod_1
*sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_2/Tensordot/concat/axis
%sequential_1/dense_2/Tensordot/concatConcatV2,sequential_1/dense_2/Tensordot/free:output:0,sequential_1/dense_2/Tensordot/axes:output:03sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_2/Tensordot/concatà
$sequential_1/dense_2/Tensordot/stackPack,sequential_1/dense_2/Tensordot/Prod:output:0.sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/stackò
(sequential_1/dense_2/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0.sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2*
(sequential_1/dense_2/Tensordot/transposeó
&sequential_1/dense_2/Tensordot/ReshapeReshape,sequential_1/dense_2/Tensordot/transpose:y:0-sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&sequential_1/dense_2/Tensordot/Reshapeò
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%sequential_1/dense_2/Tensordot/MatMul
&sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2(
&sequential_1/dense_2/Tensordot/Const_2
,sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/concat_1/axis¦
'sequential_1/dense_2/Tensordot/concat_1ConcatV20sequential_1/dense_2/Tensordot/GatherV2:output:0/sequential_1/dense_2/Tensordot/Const_2:output:05sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/concat_1ä
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:00sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2 
sequential_1/dense_2/TensordotË
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOpÛ
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_1/dense_2/BiasAdd
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
sequential_1/dense_2/ReluÕ
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02/
-sequential_1/dense_3/Tensordot/ReadVariableOp
#sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_3/Tensordot/axes
#sequential_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_3/Tensordot/free£
$sequential_1/dense_3/Tensordot/ShapeShape'sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/Shape
,sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/GatherV2/axisº
'sequential_1/dense_3/Tensordot/GatherV2GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/free:output:05sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/GatherV2¢
.sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_3/Tensordot/GatherV2_1/axisÀ
)sequential_1/dense_3/Tensordot/GatherV2_1GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/axes:output:07sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_3/Tensordot/GatherV2_1
$sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_3/Tensordot/ConstÔ
#sequential_1/dense_3/Tensordot/ProdProd0sequential_1/dense_3/Tensordot/GatherV2:output:0-sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_3/Tensordot/Prod
&sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_1Ü
%sequential_1/dense_3/Tensordot/Prod_1Prod2sequential_1/dense_3/Tensordot/GatherV2_1:output:0/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_3/Tensordot/Prod_1
*sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_3/Tensordot/concat/axis
%sequential_1/dense_3/Tensordot/concatConcatV2,sequential_1/dense_3/Tensordot/free:output:0,sequential_1/dense_3/Tensordot/axes:output:03sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_3/Tensordot/concatà
$sequential_1/dense_3/Tensordot/stackPack,sequential_1/dense_3/Tensordot/Prod:output:0.sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/stackð
(sequential_1/dense_3/Tensordot/transpose	Transpose'sequential_1/dense_2/Relu:activations:0.sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2*
(sequential_1/dense_3/Tensordot/transposeó
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_3/Tensordot/transpose:y:0-sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&sequential_1/dense_3/Tensordot/Reshapeò
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential_1/dense_3/Tensordot/MatMul
&sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_2
,sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/concat_1/axis¦
'sequential_1/dense_3/Tensordot/concat_1ConcatV20sequential_1/dense_3/Tensordot/GatherV2:output:0/sequential_1/dense_3/Tensordot/Const_2:output:05sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/concat_1ä
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:00sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
sequential_1/dense_3/TensordotË
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOpÛ
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
sequential_1/dense_3/BiasAddw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_3/dropout/Const´
dropout_3/dropout/MulMul%sequential_1/dense_3/BiasAdd:output:0 dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_3/dropout/Mul
dropout_3/dropout/ShapeShape%sequential_1/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/ShapeÖ
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_3/dropout/GreaterEqual/yê
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2 
dropout_3/dropout/GreaterEqual¡
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_3/dropout/Cast¦
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dropout_3/dropout/Mul_1
add_1AddV2)layer_normalization_2/batchnorm/add_1:z:0dropout_3/dropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_1¶
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indicesá
"layer_normalization_3/moments/meanMean	add_1:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2$
"layer_normalization_3/moments/meanË
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2,
*layer_normalization_3/moments/StopGradientí
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 21
/layer_normalization_3/moments/SquaredDifference¾
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indices
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#*
	keep_dims(2(
&layer_normalization_3/moments/variance
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½752'
%layer_normalization_3/batchnorm/add/yê
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2%
#layer_normalization_3/batchnorm/add¶
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#2'
%layer_normalization_3/batchnorm/Rsqrtà
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOpî
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_3/batchnorm/mul¿
%layer_normalization_3/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_3/batchnorm/mul_1á
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_3/batchnorm/mul_2Ô
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_3/batchnorm/ReadVariableOpê
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2%
#layer_normalization_3/batchnorm/subá
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2'
%layer_normalization_3/batchnorm/add_1Ó
IdentityIdentity)layer_normalization_3/batchnorm/add_1:z:0/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2
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
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

õ
B__inference_conv1d_layer_call_and_return_conditional_losses_139373

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
conv1d/ExpandDims¸
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim·
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1·
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
paddingSAME*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *
squeeze_dims

ýÿÿÿÿÿÿÿÿ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿR ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR 
 
_user_specified_nameinputs
Ð
 
&__inference_model_layer_call_fn_140615
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

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity¢StatefulPartitionedCallÒ
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1405402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2

d
E__inference_dropout_5_layer_call_and_return_conditional_losses_142285

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape´
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¿Z
Å
A__inference_model_layer_call_and_return_conditional_losses_140368

inputs
inputs_1'
#token_and_position_embedding_140278'
#token_and_position_embedding_140280
conv1d_140283
conv1d_140285
conv1d_1_140289
conv1d_1_140291
batch_normalization_140296
batch_normalization_140298
batch_normalization_140300
batch_normalization_140302 
batch_normalization_1_140305 
batch_normalization_1_140307 
batch_normalization_1_140309 
batch_normalization_1_140311
transformer_block_1_140315
transformer_block_1_140317
transformer_block_1_140319
transformer_block_1_140321
transformer_block_1_140323
transformer_block_1_140325
transformer_block_1_140327
transformer_block_1_140329
transformer_block_1_140331
transformer_block_1_140333
transformer_block_1_140335
transformer_block_1_140337
transformer_block_1_140339
transformer_block_1_140341
transformer_block_1_140343
transformer_block_1_140345
dense_4_140350
dense_4_140352
dense_5_140356
dense_5_140358
dense_6_140362
dense_6_140364
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢conv1d/StatefulPartitionedCall¢ conv1d_1/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall¢4token_and_position_embedding/StatefulPartitionedCall¢+transformer_block_1/StatefulPartitionedCall
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs#token_and_position_embedding_140278#token_and_position_embedding_140280*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_13934126
4token_and_position_embedding/StatefulPartitionedCallÉ
conv1d/StatefulPartitionedCallStatefulPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0conv1d_140283conv1d_140285*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿR *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_1393732 
conv1d/StatefulPartitionedCall
!average_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_1388292#
!average_pooling1d/PartitionedCallÀ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0conv1d_1_140289conv1d_1_140291*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_1394062"
 conv1d_1/StatefulPartitionedCall³
#average_pooling1d_2/PartitionedCallPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_1388592%
#average_pooling1d_2/PartitionedCall
#average_pooling1d_1/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_1388442%
#average_pooling1d_1/PartitionedCall²
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_1/PartitionedCall:output:0batch_normalization_140296batch_normalization_140298batch_normalization_140300batch_normalization_140302*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1394592-
+batch_normalization/StatefulPartitionedCallÀ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_2/PartitionedCall:output:0batch_normalization_1_140305batch_normalization_1_140307batch_normalization_1_140309batch_normalization_1_140311*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1395502/
-batch_normalization_1/StatefulPartitionedCall³
add/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:06batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_1396122
add/PartitionedCall
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0transformer_block_1_140315transformer_block_1_140317transformer_block_1_140319transformer_block_1_140321transformer_block_1_140323transformer_block_1_140325transformer_block_1_140327transformer_block_1_140329transformer_block_1_140331transformer_block_1_140333transformer_block_1_140335transformer_block_1_140337transformer_block_1_140339transformer_block_1_140341transformer_block_1_140343transformer_block_1_140345*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_1_layer_call_and_return_conditional_losses_1397692-
+transformer_block_1/StatefulPartitionedCall
flatten/PartitionedCallPartitionedCall4transformer_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1400112
flatten/PartitionedCall
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1400262
concatenate/PartitionedCall°
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_140350dense_4_140352*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1400462!
dense_4/StatefulPartitionedCall
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1400742#
!dropout_4/StatefulPartitionedCall¶
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_5_140356dense_5_140358*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1401032!
dense_5/StatefulPartitionedCall¸
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_1401312#
!dropout_5/StatefulPartitionedCall¶
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_6_140362dense_6_140364*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_1401592!
dense_6/StatefulPartitionedCall±
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ì
_input_shapesº
·:ÿÿÿÿÿÿÿÿÿR:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
§
-__inference_sequential_1_layer_call_fn_139312
dense_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCalldense_2_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_1393012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
'
_user_specified_namedense_2_input
ÙH
§
H__inference_sequential_1_layer_call_and_return_conditional_losses_142433

inputs-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource-
)dense_3_tensordot_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity¢dense_2/BiasAdd/ReadVariableOp¢ dense_2/Tensordot/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢ dense_3/Tensordot/ReadVariableOp®
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes
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
dense_2/Tensordot/Shape
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axisù
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axisÿ
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
dense_2/Tensordot/Const 
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1¨
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axisØ
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat¬
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack¨
dense_2/Tensordot/transpose	Transposeinputs!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_2/Tensordot/transpose¿
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_2/Tensordot/Reshape¾
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_2/Tensordot/MatMul
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_2/Tensordot/Const_2
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axiså
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1°
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_2/Tensordot¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp§
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_2/BiasAddt
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_2/Relu®
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02"
 dense_3/Tensordot/ReadVariableOpz
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/axes
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
dense_3/Tensordot/Shape
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/GatherV2/axisù
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axisÿ
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
dense_3/Tensordot/Const 
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1¨
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axisØ
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat¬
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stack¼
dense_3/Tensordot/transpose	Transposedense_2/Relu:activations:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@2
dense_3/Tensordot/transpose¿
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_3/Tensordot/Reshape¾
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_3/Tensordot/MatMul
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_2
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axiså
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1°
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_3/Tensordot¤
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOp§
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
dense_3/BiasAddø
IdentityIdentitydense_3/BiasAdd:output:0^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
È
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_140079

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ð	
Ü
C__inference_dense_4_layer_call_and_return_conditional_losses_140046

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	è@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿè::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
 
_user_specified_nameinputs
È
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_142290

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
è

Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_139570

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOp
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
batchnorm/add_1ß
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
Æ
§
4__inference_batch_normalization_layer_call_fn_141657

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_1394792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
âÅ
Ý0
"__inference__traced_restore_143016
file_prefix"
assignvariableop_conv1d_kernel"
assignvariableop_1_conv1d_bias&
"assignvariableop_2_conv1d_1_kernel$
 assignvariableop_3_conv1d_1_bias0
,assignvariableop_4_batch_normalization_gamma/
+assignvariableop_5_batch_normalization_beta6
2assignvariableop_6_batch_normalization_moving_mean:
6assignvariableop_7_batch_normalization_moving_variance2
.assignvariableop_8_batch_normalization_1_gamma1
-assignvariableop_9_batch_normalization_1_beta9
5assignvariableop_10_batch_normalization_1_moving_mean=
9assignvariableop_11_batch_normalization_1_moving_variance&
"assignvariableop_12_dense_4_kernel$
 assignvariableop_13_dense_4_bias&
"assignvariableop_14_dense_5_kernel$
 assignvariableop_15_dense_5_bias&
"assignvariableop_16_dense_6_kernel$
 assignvariableop_17_dense_6_bias
assignvariableop_18_decay%
!assignvariableop_19_learning_rate 
assignvariableop_20_momentum 
assignvariableop_21_sgd_iterI
Eassignvariableop_22_token_and_position_embedding_embedding_embeddingsK
Gassignvariableop_23_token_and_position_embedding_embedding_1_embeddingsO
Kassignvariableop_24_transformer_block_1_multi_head_attention_1_query_kernelM
Iassignvariableop_25_transformer_block_1_multi_head_attention_1_query_biasM
Iassignvariableop_26_transformer_block_1_multi_head_attention_1_key_kernelK
Gassignvariableop_27_transformer_block_1_multi_head_attention_1_key_biasO
Kassignvariableop_28_transformer_block_1_multi_head_attention_1_value_kernelM
Iassignvariableop_29_transformer_block_1_multi_head_attention_1_value_biasZ
Vassignvariableop_30_transformer_block_1_multi_head_attention_1_attention_output_kernelX
Tassignvariableop_31_transformer_block_1_multi_head_attention_1_attention_output_bias&
"assignvariableop_32_dense_2_kernel$
 assignvariableop_33_dense_2_bias&
"assignvariableop_34_dense_3_kernel$
 assignvariableop_35_dense_3_biasG
Cassignvariableop_36_transformer_block_1_layer_normalization_2_gammaF
Bassignvariableop_37_transformer_block_1_layer_normalization_2_betaG
Cassignvariableop_38_transformer_block_1_layer_normalization_3_gammaF
Bassignvariableop_39_transformer_block_1_layer_normalization_3_beta
assignvariableop_40_total
assignvariableop_41_count2
.assignvariableop_42_sgd_conv1d_kernel_momentum0
,assignvariableop_43_sgd_conv1d_bias_momentum4
0assignvariableop_44_sgd_conv1d_1_kernel_momentum2
.assignvariableop_45_sgd_conv1d_1_bias_momentum>
:assignvariableop_46_sgd_batch_normalization_gamma_momentum=
9assignvariableop_47_sgd_batch_normalization_beta_momentum@
<assignvariableop_48_sgd_batch_normalization_1_gamma_momentum?
;assignvariableop_49_sgd_batch_normalization_1_beta_momentum3
/assignvariableop_50_sgd_dense_4_kernel_momentum1
-assignvariableop_51_sgd_dense_4_bias_momentum3
/assignvariableop_52_sgd_dense_5_kernel_momentum1
-assignvariableop_53_sgd_dense_5_bias_momentum3
/assignvariableop_54_sgd_dense_6_kernel_momentum1
-assignvariableop_55_sgd_dense_6_bias_momentumV
Rassignvariableop_56_sgd_token_and_position_embedding_embedding_embeddings_momentumX
Tassignvariableop_57_sgd_token_and_position_embedding_embedding_1_embeddings_momentum\
Xassignvariableop_58_sgd_transformer_block_1_multi_head_attention_1_query_kernel_momentumZ
Vassignvariableop_59_sgd_transformer_block_1_multi_head_attention_1_query_bias_momentumZ
Vassignvariableop_60_sgd_transformer_block_1_multi_head_attention_1_key_kernel_momentumX
Tassignvariableop_61_sgd_transformer_block_1_multi_head_attention_1_key_bias_momentum\
Xassignvariableop_62_sgd_transformer_block_1_multi_head_attention_1_value_kernel_momentumZ
Vassignvariableop_63_sgd_transformer_block_1_multi_head_attention_1_value_bias_momentumg
cassignvariableop_64_sgd_transformer_block_1_multi_head_attention_1_attention_output_kernel_momentume
aassignvariableop_65_sgd_transformer_block_1_multi_head_attention_1_attention_output_bias_momentum3
/assignvariableop_66_sgd_dense_2_kernel_momentum1
-assignvariableop_67_sgd_dense_2_bias_momentum3
/assignvariableop_68_sgd_dense_3_kernel_momentum1
-assignvariableop_69_sgd_dense_3_bias_momentumT
Passignvariableop_70_sgd_transformer_block_1_layer_normalization_2_gamma_momentumS
Oassignvariableop_71_sgd_transformer_block_1_layer_normalization_2_beta_momentumT
Passignvariableop_72_sgd_transformer_block_1_layer_normalization_3_gamma_momentumS
Oassignvariableop_73_sgd_transformer_block_1_layer_normalization_3_beta_momentum
identity_75¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_8¢AssignVariableOp_9é%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*õ$
valueë$Bè$KB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names§
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*«
value¡BKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¥
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Â
_output_shapes¯
¬:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Y
dtypesO
M2K	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv1d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4±
AssignVariableOp_4AssignVariableOp,assignvariableop_4_batch_normalization_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5°
AssignVariableOp_5AssignVariableOp+assignvariableop_5_batch_normalization_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6·
AssignVariableOp_6AssignVariableOp2assignvariableop_6_batch_normalization_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7»
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8³
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9²
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10½
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Á
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ª
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_4_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¨
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_4_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ª
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_5_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¨
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_5_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ª
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_6_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¨
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_6_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¡
AssignVariableOp_18AssignVariableOpassignvariableop_18_decayIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19©
AssignVariableOp_19AssignVariableOp!assignvariableop_19_learning_rateIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¤
AssignVariableOp_20AssignVariableOpassignvariableop_20_momentumIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_21¤
AssignVariableOp_21AssignVariableOpassignvariableop_21_sgd_iterIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Í
AssignVariableOp_22AssignVariableOpEassignvariableop_22_token_and_position_embedding_embedding_embeddingsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ï
AssignVariableOp_23AssignVariableOpGassignvariableop_23_token_and_position_embedding_embedding_1_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ó
AssignVariableOp_24AssignVariableOpKassignvariableop_24_transformer_block_1_multi_head_attention_1_query_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ñ
AssignVariableOp_25AssignVariableOpIassignvariableop_25_transformer_block_1_multi_head_attention_1_query_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ñ
AssignVariableOp_26AssignVariableOpIassignvariableop_26_transformer_block_1_multi_head_attention_1_key_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ï
AssignVariableOp_27AssignVariableOpGassignvariableop_27_transformer_block_1_multi_head_attention_1_key_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ó
AssignVariableOp_28AssignVariableOpKassignvariableop_28_transformer_block_1_multi_head_attention_1_value_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ñ
AssignVariableOp_29AssignVariableOpIassignvariableop_29_transformer_block_1_multi_head_attention_1_value_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Þ
AssignVariableOp_30AssignVariableOpVassignvariableop_30_transformer_block_1_multi_head_attention_1_attention_output_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ü
AssignVariableOp_31AssignVariableOpTassignvariableop_31_transformer_block_1_multi_head_attention_1_attention_output_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32ª
AssignVariableOp_32AssignVariableOp"assignvariableop_32_dense_2_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33¨
AssignVariableOp_33AssignVariableOp assignvariableop_33_dense_2_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34ª
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_3_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¨
AssignVariableOp_35AssignVariableOp assignvariableop_35_dense_3_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ë
AssignVariableOp_36AssignVariableOpCassignvariableop_36_transformer_block_1_layer_normalization_2_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ê
AssignVariableOp_37AssignVariableOpBassignvariableop_37_transformer_block_1_layer_normalization_2_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ë
AssignVariableOp_38AssignVariableOpCassignvariableop_38_transformer_block_1_layer_normalization_3_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Ê
AssignVariableOp_39AssignVariableOpBassignvariableop_39_transformer_block_1_layer_normalization_3_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¡
AssignVariableOp_40AssignVariableOpassignvariableop_40_totalIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41¡
AssignVariableOp_41AssignVariableOpassignvariableop_41_countIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¶
AssignVariableOp_42AssignVariableOp.assignvariableop_42_sgd_conv1d_kernel_momentumIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43´
AssignVariableOp_43AssignVariableOp,assignvariableop_43_sgd_conv1d_bias_momentumIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¸
AssignVariableOp_44AssignVariableOp0assignvariableop_44_sgd_conv1d_1_kernel_momentumIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¶
AssignVariableOp_45AssignVariableOp.assignvariableop_45_sgd_conv1d_1_bias_momentumIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Â
AssignVariableOp_46AssignVariableOp:assignvariableop_46_sgd_batch_normalization_gamma_momentumIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Á
AssignVariableOp_47AssignVariableOp9assignvariableop_47_sgd_batch_normalization_beta_momentumIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Ä
AssignVariableOp_48AssignVariableOp<assignvariableop_48_sgd_batch_normalization_1_gamma_momentumIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Ã
AssignVariableOp_49AssignVariableOp;assignvariableop_49_sgd_batch_normalization_1_beta_momentumIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50·
AssignVariableOp_50AssignVariableOp/assignvariableop_50_sgd_dense_4_kernel_momentumIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51µ
AssignVariableOp_51AssignVariableOp-assignvariableop_51_sgd_dense_4_bias_momentumIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52·
AssignVariableOp_52AssignVariableOp/assignvariableop_52_sgd_dense_5_kernel_momentumIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53µ
AssignVariableOp_53AssignVariableOp-assignvariableop_53_sgd_dense_5_bias_momentumIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54·
AssignVariableOp_54AssignVariableOp/assignvariableop_54_sgd_dense_6_kernel_momentumIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55µ
AssignVariableOp_55AssignVariableOp-assignvariableop_55_sgd_dense_6_bias_momentumIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Ú
AssignVariableOp_56AssignVariableOpRassignvariableop_56_sgd_token_and_position_embedding_embedding_embeddings_momentumIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57Ü
AssignVariableOp_57AssignVariableOpTassignvariableop_57_sgd_token_and_position_embedding_embedding_1_embeddings_momentumIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58à
AssignVariableOp_58AssignVariableOpXassignvariableop_58_sgd_transformer_block_1_multi_head_attention_1_query_kernel_momentumIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Þ
AssignVariableOp_59AssignVariableOpVassignvariableop_59_sgd_transformer_block_1_multi_head_attention_1_query_bias_momentumIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Þ
AssignVariableOp_60AssignVariableOpVassignvariableop_60_sgd_transformer_block_1_multi_head_attention_1_key_kernel_momentumIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Ü
AssignVariableOp_61AssignVariableOpTassignvariableop_61_sgd_transformer_block_1_multi_head_attention_1_key_bias_momentumIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62à
AssignVariableOp_62AssignVariableOpXassignvariableop_62_sgd_transformer_block_1_multi_head_attention_1_value_kernel_momentumIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Þ
AssignVariableOp_63AssignVariableOpVassignvariableop_63_sgd_transformer_block_1_multi_head_attention_1_value_bias_momentumIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64ë
AssignVariableOp_64AssignVariableOpcassignvariableop_64_sgd_transformer_block_1_multi_head_attention_1_attention_output_kernel_momentumIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65é
AssignVariableOp_65AssignVariableOpaassignvariableop_65_sgd_transformer_block_1_multi_head_attention_1_attention_output_bias_momentumIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66·
AssignVariableOp_66AssignVariableOp/assignvariableop_66_sgd_dense_2_kernel_momentumIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67µ
AssignVariableOp_67AssignVariableOp-assignvariableop_67_sgd_dense_2_bias_momentumIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68·
AssignVariableOp_68AssignVariableOp/assignvariableop_68_sgd_dense_3_kernel_momentumIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69µ
AssignVariableOp_69AssignVariableOp-assignvariableop_69_sgd_dense_3_bias_momentumIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Ø
AssignVariableOp_70AssignVariableOpPassignvariableop_70_sgd_transformer_block_1_layer_normalization_2_gamma_momentumIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71×
AssignVariableOp_71AssignVariableOpOassignvariableop_71_sgd_transformer_block_1_layer_normalization_2_beta_momentumIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72Ø
AssignVariableOp_72AssignVariableOpPassignvariableop_72_sgd_transformer_block_1_layer_normalization_3_gamma_momentumIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73×
AssignVariableOp_73AssignVariableOpOassignvariableop_73_sgd_transformer_block_1_layer_normalization_3_beta_momentumIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_739
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpº
Identity_74Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_74­
Identity_75IdentityIdentity_74:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_75"#
identity_75Identity_75:output:0*¿
_input_shapes­
ª: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
½
k
?__inference_add_layer_call_and_return_conditional_losses_141827
inputs_0
inputs_1
identity]
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿ# :ÿÿÿÿÿÿÿÿÿ# :U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
"
_user_specified_name
inputs/1

F
*__inference_dropout_4_layer_call_fn_142253

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_1400792
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Ü
C__inference_dense_6_layer_call_and_return_conditional_losses_140159

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ð

à
4__inference_transformer_block_1_layer_call_fn_142182

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
identity¢StatefulPartitionedCallÁ
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
:ÿÿÿÿÿÿÿÿÿ# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_transformer_block_1_layer_call_and_return_conditional_losses_1398962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:ÿÿÿÿÿÿÿÿÿ# ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
í
}
(__inference_dense_3_layer_call_fn_142538

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1392262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ#@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ#@
 
_user_specified_nameinputs
È
©
6__inference_batch_normalization_1_layer_call_fn_141808

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1395502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs

D
(__inference_flatten_layer_call_fn_142193

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1400112
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ# :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs
È
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_140136

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
À
s
G__inference_concatenate_layer_call_and_return_conditional_losses_142200
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿà:ÿÿÿÿÿÿÿÿÿ:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
¤
X
,__inference_concatenate_layer_call_fn_142206
inputs_0
inputs_1
identityÖ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_1400262
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿà:ÿÿÿÿÿÿÿÿÿ:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ó0
È
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_141693

inputs
assignmovingavg_141668
assignmovingavg_1_141674)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢AssignMovingAvg/ReadVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices¶
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Ì
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/141668*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_141668*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpñ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/141668*
_output_shapes
: 2
AssignMovingAvg/subè
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/141668*
_output_shapes
: 2
AssignMovingAvg/mul¯
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_141668AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/141668*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpÒ
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/141674*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_141674*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpû
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/141674*
_output_shapes
: 2
AssignMovingAvg_1/subò
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/141674*
_output_shapes
: 2
AssignMovingAvg_1/mul»
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_141674AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/141674*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
batchnorm/add_1À
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
³
_
C__inference_flatten_layer_call_and_return_conditional_losses_142188

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ# :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ# 
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*è
serving_defaultÔ
<
input_11
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿR
;
input_20
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿ;
dense_60
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ïù
G
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
±_default_save_signature
²__call__
+³&call_and_return_all_conditional_losses"ØA
_tf_keras_network¼A{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["token_and_position_embedding", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["average_pooling1d", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_1", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_2", "inbound_nodes": [[["token_and_position_embedding", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["average_pooling1d_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["average_pooling1d_2", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["batch_normalization", 0, 0, {}], ["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_1", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["transformer_block_1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["flatten", 0, 0, {}], ["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dropout_4", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_5", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dropout_5", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["dense_6", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 10500]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10500]}, {"class_name": "TensorShape", "items": [null, 8]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.00020000000949949026, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
ñ"î
_tf_keras_input_layerÎ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
å
	token_emb
pos_emb
regularization_losses
	variables
trainable_variables
	keras_api
´__call__
+µ&call_and_return_all_conditional_losses"¸
_tf_keras_layer{"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
å	

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"¾
_tf_keras_layer¤{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500, 32]}}

&regularization_losses
'	variables
(trainable_variables
)	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses"ô
_tf_keras_layerÚ{"class_name": "AveragePooling1D", "name": "average_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ç	

*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
º__call__
+»&call_and_return_all_conditional_losses"À
_tf_keras_layer¦{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 350, 32]}}

0regularization_losses
1	variables
2trainable_variables
3	keras_api
¼__call__
+½&call_and_return_all_conditional_losses"ø
_tf_keras_layerÞ{"class_name": "AveragePooling1D", "name": "average_pooling1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}

4regularization_losses
5	variables
6trainable_variables
7	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"class_name": "AveragePooling1D", "name": "average_pooling1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
´	
8axis
	9gamma
:beta
;moving_mean
<moving_variance
=regularization_losses
>	variables
?trainable_variables
@	keras_api
À__call__
+Á&call_and_return_all_conditional_losses"Þ
_tf_keras_layerÄ{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
¸	
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses"â
_tf_keras_layerÈ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
¯
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "Add", "name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 35, 32]}, {"class_name": "TensorShape", "items": [null, 35, 32]}]}

Natt
Offn
P
layernorm1
Q
layernorm2
Rdropout1
Sdropout2
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
Æ__call__
+Ç&call_and_return_all_conditional_losses"¥
_tf_keras_layer{"class_name": "TransformerBlock", "name": "transformer_block_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
ä
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
È__call__
+É&call_and_return_all_conditional_losses"Ó
_tf_keras_layer¹{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
é"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
Ì
\regularization_losses
]	variables
^trainable_variables
_	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses"»
_tf_keras_layer¡{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1120]}, {"class_name": "TensorShape", "items": [null, 8]}]}
ö

`kernel
abias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1128]}}
ç
fregularization_losses
g	variables
htrainable_variables
i	keras_api
Î__call__
+Ï&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
ò

jkernel
kbias
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses"Ë
_tf_keras_layer±{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ç
pregularization_losses
q	variables
rtrainable_variables
s	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
ó

tkernel
ubias
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
Ô__call__
+Õ&call_and_return_all_conditional_losses"Ì
_tf_keras_layer²{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ù
	zdecay
{learning_rate
|momentum
}iter momentum!momentum*momentum+momentum9momentum:momentumBmomentumCmomentum`momentumamomentumjmomentumkmomentumtmomentumumomentum~momentummomentum momentum¡momentum¢momentum£momentum¤momentum¥momentum¦momentum§momentum¨momentum©momentumªmomentum«momentum¬momentum­momentum®momentum¯momentum°"
	optimizer
 "
trackable_list_wrapper
Æ
~0
1
 2
!3
*4
+5
96
:7
;8
<9
B10
C11
D12
E13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
`30
a31
j32
k33
t34
u35"
trackable_list_wrapper
¦
~0
1
 2
!3
*4
+5
96
:7
B8
C9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
`26
a27
j28
k29
t30
u31"
trackable_list_wrapper
Ó
regularization_losses
metrics
	variables
trainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
²__call__
±_default_save_signature
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
-
Öserving_default"
signature_map
°
~
embeddings
regularization_losses
	variables
trainable_variables
	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses"
_tf_keras_layerñ{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500]}}
±

embeddings
regularization_losses
	variables
trainable_variables
	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses"
_tf_keras_layerò{"class_name": "Embedding", "name": "embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 10500, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
 "
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
µ
regularization_losses
metrics
	variables
trainable_variables
layer_metrics
 layer_regularization_losses
 non_trainable_variables
¡layers
´__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
#:!  2conv1d/kernel
: 2conv1d/bias
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
µ
"regularization_losses
¢metrics
#	variables
$trainable_variables
£layer_metrics
 ¤layer_regularization_losses
¥non_trainable_variables
¦layers
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
&regularization_losses
§metrics
'	variables
(trainable_variables
¨layer_metrics
 ©layer_regularization_losses
ªnon_trainable_variables
«layers
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
%:#	  2conv1d_1/kernel
: 2conv1d_1/bias
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
µ
,regularization_losses
¬metrics
-	variables
.trainable_variables
­layer_metrics
 ®layer_regularization_losses
¯non_trainable_variables
°layers
º__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
0regularization_losses
±metrics
1	variables
2trainable_variables
²layer_metrics
 ³layer_regularization_losses
´non_trainable_variables
µlayers
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
4regularization_losses
¶metrics
5	variables
6trainable_variables
·layer_metrics
 ¸layer_regularization_losses
¹non_trainable_variables
ºlayers
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
<
90
:1
;2
<3"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
µ
=regularization_losses
»metrics
>	variables
?trainable_variables
¼layer_metrics
 ½layer_regularization_losses
¾non_trainable_variables
¿layers
À__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_1/gamma
(:& 2batch_normalization_1/beta
1:/  (2!batch_normalization_1/moving_mean
5:3  (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
µ
Fregularization_losses
Àmetrics
G	variables
Htrainable_variables
Álayer_metrics
 Âlayer_regularization_losses
Ãnon_trainable_variables
Älayers
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Jregularization_losses
Åmetrics
K	variables
Ltrainable_variables
Ælayer_metrics
 Çlayer_regularization_losses
Ènon_trainable_variables
Élayers
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object

Ê_query_dense
Ë
_key_dense
Ì_value_dense
Í_softmax
Î_dropout_layer
Ï_output_dense
Ðregularization_losses
Ñ	variables
Òtrainable_variables
Ó	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses"
_tf_keras_layerê{"class_name": "MultiHeadAttention", "name": "multi_head_attention_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_1", "trainable": true, "dtype": "float32", "num_heads": 1, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
©
Ôlayer_with_weights-0
Ôlayer-0
Õlayer_with_weights-1
Õlayer-1
Öregularization_losses
×	variables
Øtrainable_variables
Ù	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses"Â
_tf_keras_sequential£{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_2_input"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_2_input"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
ê
	Úaxis

gamma
	beta
Ûregularization_losses
Ü	variables
Ýtrainable_variables
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses"³
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
ê
	ßaxis

gamma
	beta
àregularization_losses
á	variables
âtrainable_variables
ã	keras_api
á__call__
+â&call_and_return_all_conditional_losses"³
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_3", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
ë
äregularization_losses
å	variables
ætrainable_variables
ç	keras_api
ã__call__
+ä&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
ë
èregularization_losses
é	variables
êtrainable_variables
ë	keras_api
å__call__
+æ&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
 "
trackable_list_wrapper
¦
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
¦
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
µ
Tregularization_losses
ìmetrics
U	variables
Vtrainable_variables
ílayer_metrics
 îlayer_regularization_losses
ïnon_trainable_variables
ðlayers
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Xregularization_losses
ñmetrics
Y	variables
Ztrainable_variables
òlayer_metrics
 ólayer_regularization_losses
ônon_trainable_variables
õlayers
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
\regularization_losses
ömetrics
]	variables
^trainable_variables
÷layer_metrics
 ølayer_regularization_losses
ùnon_trainable_variables
úlayers
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
!:	è@2dense_4/kernel
:@2dense_4/bias
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
µ
bregularization_losses
ûmetrics
c	variables
dtrainable_variables
ülayer_metrics
 ýlayer_regularization_losses
þnon_trainable_variables
ÿlayers
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
fregularization_losses
metrics
g	variables
htrainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
Î__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
 :@@2dense_5/kernel
:@2dense_5/bias
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
µ
lregularization_losses
metrics
m	variables
ntrainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
pregularization_losses
metrics
q	variables
rtrainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_6/kernel
:2dense_6/bias
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
µ
vregularization_losses
metrics
w	variables
xtrainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
Ô__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
C:A 21token_and_position_embedding/embedding/embeddings
F:D	R 23token_and_position_embedding/embedding_1/embeddings
M:K  27transformer_block_1/multi_head_attention_1/query/kernel
G:E 25transformer_block_1/multi_head_attention_1/query/bias
K:I  25transformer_block_1/multi_head_attention_1/key/kernel
E:C 23transformer_block_1/multi_head_attention_1/key/bias
M:K  27transformer_block_1/multi_head_attention_1/value/kernel
G:E 25transformer_block_1/multi_head_attention_1/value/bias
X:V  2Btransformer_block_1/multi_head_attention_1/attention_output/kernel
N:L 2@transformer_block_1/multi_head_attention_1/attention_output/bias
 : @2dense_2/kernel
:@2dense_2/bias
 :@ 2dense_3/kernel
: 2dense_3/bias
=:; 2/transformer_block_1/layer_normalization_2/gamma
<:: 2.transformer_block_1/layer_normalization_2/beta
=:; 2/transformer_block_1/layer_normalization_3/gamma
<:: 2.transformer_block_1/layer_normalization_3/beta
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
;0
<1
D2
E3"
trackable_list_wrapper
®
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
12
13
14
15
16
17
18"
trackable_list_wrapper
 "
trackable_list_wrapper
'
~0"
trackable_list_wrapper
'
~0"
trackable_list_wrapper
¸
regularization_losses
metrics
	variables
trainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
¸
regularization_losses
metrics
	variables
trainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
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
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
D0
E1"
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
Ë
partial_output_shape
 full_output_shape
kernel
	bias
¡regularization_losses
¢	variables
£trainable_variables
¤	keras_api
ç__call__
+è&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
Ç
¥partial_output_shape
¦full_output_shape
kernel
	bias
§regularization_losses
¨	variables
©trainable_variables
ª	keras_api
é__call__
+ê&call_and_return_all_conditional_losses"ç
_tf_keras_layerÍ{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
Ë
«partial_output_shape
¬full_output_shape
kernel
	bias
­regularization_losses
®	variables
¯trainable_variables
°	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
ë
±regularization_losses
²	variables
³trainable_variables
´	keras_api
í__call__
+î&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
ç
µregularization_losses
¶	variables
·trainable_variables
¸	keras_api
ï__call__
+ð&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
à
¹partial_output_shape
ºfull_output_shape
kernel
	bias
»regularization_losses
¼	variables
½trainable_variables
¾	keras_api
ñ__call__
+ò&call_and_return_all_conditional_losses"
_tf_keras_layeræ{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 32], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 1, 32]}}
 "
trackable_list_wrapper
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
`
0
1
2
3
4
5
6
7"
trackable_list_wrapper
¸
Ðregularization_losses
¿metrics
Ñ	variables
Òtrainable_variables
Àlayer_metrics
 Álayer_regularization_losses
Ânon_trainable_variables
Ãlayers
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
ü
kernel
	bias
Äregularization_losses
Å	variables
Ætrainable_variables
Ç	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
þ
kernel
	bias
Èregularization_losses
É	variables
Êtrainable_variables
Ë	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 64]}}
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
¸
Öregularization_losses
Ìmetrics
×	variables
Øtrainable_variables
Ílayer_metrics
 Îlayer_regularization_losses
Ïnon_trainable_variables
Ðlayers
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
Ûregularization_losses
Ñmetrics
Ü	variables
Ýtrainable_variables
Òlayer_metrics
 Ólayer_regularization_losses
Ônon_trainable_variables
Õlayers
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
àregularization_losses
Ömetrics
á	variables
âtrainable_variables
×layer_metrics
 Ølayer_regularization_losses
Ùnon_trainable_variables
Úlayers
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
äregularization_losses
Ûmetrics
å	variables
ætrainable_variables
Ülayer_metrics
 Ýlayer_regularization_losses
Þnon_trainable_variables
ßlayers
ã__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
èregularization_losses
àmetrics
é	variables
êtrainable_variables
álayer_metrics
 âlayer_regularization_losses
ãnon_trainable_variables
älayers
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
N0
O1
P2
Q3
R4
S5"
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
¿

åtotal

æcount
ç	variables
è	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
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
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
¡regularization_losses
émetrics
¢	variables
£trainable_variables
êlayer_metrics
 ëlayer_regularization_losses
ìnon_trainable_variables
ílayers
ç__call__
+è&call_and_return_all_conditional_losses
'è"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
§regularization_losses
îmetrics
¨	variables
©trainable_variables
ïlayer_metrics
 ðlayer_regularization_losses
ñnon_trainable_variables
òlayers
é__call__
+ê&call_and_return_all_conditional_losses
'ê"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
­regularization_losses
ómetrics
®	variables
¯trainable_variables
ôlayer_metrics
 õlayer_regularization_losses
önon_trainable_variables
÷layers
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
±regularization_losses
ømetrics
²	variables
³trainable_variables
ùlayer_metrics
 úlayer_regularization_losses
ûnon_trainable_variables
ülayers
í__call__
+î&call_and_return_all_conditional_losses
'î"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
µregularization_losses
ýmetrics
¶	variables
·trainable_variables
þlayer_metrics
 ÿlayer_regularization_losses
non_trainable_variables
layers
ï__call__
+ð&call_and_return_all_conditional_losses
'ð"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
»regularization_losses
metrics
¼	variables
½trainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
ñ__call__
+ò&call_and_return_all_conditional_losses
'ò"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
P
Ê0
Ë1
Ì2
Í3
Î4
Ï5"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
Äregularization_losses
metrics
Å	variables
Ætrainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
Èregularization_losses
metrics
É	variables
Êtrainable_variables
layer_metrics
 layer_regularization_losses
non_trainable_variables
layers
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ô0
Õ1"
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
:  (2total
:  (2count
0
å0
æ1"
trackable_list_wrapper
.
ç	variables"
_generic_user_object
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
 "
trackable_list_wrapper
.:,  2SGD/conv1d/kernel/momentum
$:" 2SGD/conv1d/bias/momentum
0:.	  2SGD/conv1d_1/kernel/momentum
&:$ 2SGD/conv1d_1/bias/momentum
2:0 2&SGD/batch_normalization/gamma/momentum
1:/ 2%SGD/batch_normalization/beta/momentum
4:2 2(SGD/batch_normalization_1/gamma/momentum
3:1 2'SGD/batch_normalization_1/beta/momentum
,:*	è@2SGD/dense_4/kernel/momentum
%:#@2SGD/dense_4/bias/momentum
+:)@@2SGD/dense_5/kernel/momentum
%:#@2SGD/dense_5/bias/momentum
+:)@2SGD/dense_6/kernel/momentum
%:#2SGD/dense_6/bias/momentum
N:L 2>SGD/token_and_position_embedding/embedding/embeddings/momentum
Q:O	R 2@SGD/token_and_position_embedding/embedding_1/embeddings/momentum
X:V  2DSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentum
R:P 2BSGD/transformer_block_1/multi_head_attention_1/query/bias/momentum
V:T  2BSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentum
P:N 2@SGD/transformer_block_1/multi_head_attention_1/key/bias/momentum
X:V  2DSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentum
R:P 2BSGD/transformer_block_1/multi_head_attention_1/value/bias/momentum
c:a  2OSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentum
Y:W 2MSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentum
+:) @2SGD/dense_2/kernel/momentum
%:#@2SGD/dense_2/bias/momentum
+:)@ 2SGD/dense_3/kernel/momentum
%:# 2SGD/dense_3/bias/momentum
H:F 2<SGD/transformer_block_1/layer_normalization_2/gamma/momentum
G:E 2;SGD/transformer_block_1/layer_normalization_2/beta/momentum
H:F 2<SGD/transformer_block_1/layer_normalization_3/gamma/momentum
G:E 2;SGD/transformer_block_1/layer_normalization_3/beta/momentum
2
!__inference__wrapped_model_138820ß
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *O¢L
JG
"
input_1ÿÿÿÿÿÿÿÿÿR
!
input_2ÿÿÿÿÿÿÿÿÿ
æ2ã
&__inference_model_layer_call_fn_141410
&__inference_model_layer_call_fn_140615
&__inference_model_layer_call_fn_141332
&__inference_model_layer_call_fn_140443À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
A__inference_model_layer_call_and_return_conditional_losses_141011
A__inference_model_layer_call_and_return_conditional_losses_140176
A__inference_model_layer_call_and_return_conditional_losses_140270
A__inference_model_layer_call_and_return_conditional_losses_141254À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
â2ß
=__inference_token_and_position_embedding_layer_call_fn_141443
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ý2ú
X__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_141434
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_conv1d_layer_call_fn_141468¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_conv1d_layer_call_and_return_conditional_losses_141459¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
2__inference_average_pooling1d_layer_call_fn_138835Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¨2¥
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_138829Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ó2Ð
)__inference_conv1d_1_layer_call_fn_141493¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv1d_1_layer_call_and_return_conditional_losses_141484¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
4__inference_average_pooling1d_1_layer_call_fn_138850Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª2§
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_138844Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
4__inference_average_pooling1d_2_layer_call_fn_138865Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª2§
O__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_138859Ó
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
4__inference_batch_normalization_layer_call_fn_141575
4__inference_batch_normalization_layer_call_fn_141562
4__inference_batch_normalization_layer_call_fn_141644
4__inference_batch_normalization_layer_call_fn_141657´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þ2û
O__inference_batch_normalization_layer_call_and_return_conditional_losses_141529
O__inference_batch_normalization_layer_call_and_return_conditional_losses_141549
O__inference_batch_normalization_layer_call_and_return_conditional_losses_141611
O__inference_batch_normalization_layer_call_and_return_conditional_losses_141631´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
6__inference_batch_normalization_1_layer_call_fn_141821
6__inference_batch_normalization_1_layer_call_fn_141739
6__inference_batch_normalization_1_layer_call_fn_141726
6__inference_batch_normalization_1_layer_call_fn_141808´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_141775
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_141795
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_141713
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_141693´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Î2Ë
$__inference_add_layer_call_fn_141833¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_add_layer_call_and_return_conditional_losses_141827¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¢2
4__inference_transformer_block_1_layer_call_fn_142182
4__inference_transformer_block_1_layer_call_fn_142145°
§²£
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
O__inference_transformer_block_1_layer_call_and_return_conditional_losses_142108
O__inference_transformer_block_1_layer_call_and_return_conditional_losses_141981°
§²£
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
(__inference_flatten_layer_call_fn_142193¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_flatten_layer_call_and_return_conditional_losses_142188¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_concatenate_layer_call_fn_142206¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_concatenate_layer_call_and_return_conditional_losses_142200¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_4_layer_call_fn_142226¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_4_layer_call_and_return_conditional_losses_142217¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
*__inference_dropout_4_layer_call_fn_142253
*__inference_dropout_4_layer_call_fn_142248´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_4_layer_call_and_return_conditional_losses_142238
E__inference_dropout_4_layer_call_and_return_conditional_losses_142243´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
(__inference_dense_5_layer_call_fn_142273¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_5_layer_call_and_return_conditional_losses_142264¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
*__inference_dropout_5_layer_call_fn_142295
*__inference_dropout_5_layer_call_fn_142300´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_5_layer_call_and_return_conditional_losses_142290
E__inference_dropout_5_layer_call_and_return_conditional_losses_142285´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
(__inference_dense_6_layer_call_fn_142319¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_6_layer_call_and_return_conditional_losses_142310¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÒBÏ
$__inference_signature_wrapper_140701input_1input_2"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2ÿü
ó²ï
FullArgSpece
args]Z
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
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿü
ó²ï
FullArgSpece
args]Z
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
defaults

 

 
p 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
-__inference_sequential_1_layer_call_fn_142459
-__inference_sequential_1_layer_call_fn_139285
-__inference_sequential_1_layer_call_fn_142446
-__inference_sequential_1_layer_call_fn_139312À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_sequential_1_layer_call_and_return_conditional_losses_139243
H__inference_sequential_1_layer_call_and_return_conditional_losses_142376
H__inference_sequential_1_layer_call_and_return_conditional_losses_142433
H__inference_sequential_1_layer_call_and_return_conditional_losses_139257À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ2²¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ2²¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
º2·´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_2_layer_call_fn_142499¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_2_layer_call_and_return_conditional_losses_142490¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_3_layer_call_fn_142538¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_3_layer_call_and_return_conditional_losses_142529¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ê
!__inference__wrapped_model_138820Ä4~ !*+<9;:EBDC`ajktuY¢V
O¢L
JG
"
input_1ÿÿÿÿÿÿÿÿÿR
!
input_2ÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_6!
dense_6ÿÿÿÿÿÿÿÿÿÓ
?__inference_add_layer_call_and_return_conditional_losses_141827b¢_
X¢U
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿ# 
&#
inputs/1ÿÿÿÿÿÿÿÿÿ# 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 «
$__inference_add_layer_call_fn_141833b¢_
X¢U
SP
&#
inputs/0ÿÿÿÿÿÿÿÿÿ# 
&#
inputs/1ÿÿÿÿÿÿÿÿÿ# 
ª "ÿÿÿÿÿÿÿÿÿ# Ø
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_138844E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
4__inference_average_pooling1d_1_layer_call_fn_138850wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿØ
O__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_138859E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
4__inference_average_pooling1d_2_layer_call_fn_138865wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÖ
M__inference_average_pooling1d_layer_call_and_return_conditional_losses_138829E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ­
2__inference_average_pooling1d_layer_call_fn_138835wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÑ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_141693|DEBC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ñ
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_141713|EBDC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¿
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_141775jDEBC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ¿
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_141795jEBDC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ©
6__inference_batch_normalization_1_layer_call_fn_141726oDEBC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ©
6__inference_batch_normalization_1_layer_call_fn_141739oEBDC@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
6__inference_batch_normalization_1_layer_call_fn_141808]DEBC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª "ÿÿÿÿÿÿÿÿÿ# 
6__inference_batch_normalization_1_layer_call_fn_141821]EBDC7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª "ÿÿÿÿÿÿÿÿÿ# Ï
O__inference_batch_normalization_layer_call_and_return_conditional_losses_141529|;<9:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ï
O__inference_batch_normalization_layer_call_and_return_conditional_losses_141549|<9;:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ½
O__inference_batch_normalization_layer_call_and_return_conditional_losses_141611j;<9:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ½
O__inference_batch_normalization_layer_call_and_return_conditional_losses_141631j<9;:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 §
4__inference_batch_normalization_layer_call_fn_141562o;<9:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ §
4__inference_batch_normalization_layer_call_fn_141575o<9;:@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
4__inference_batch_normalization_layer_call_fn_141644];<9:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª "ÿÿÿÿÿÿÿÿÿ# 
4__inference_batch_normalization_layer_call_fn_141657]<9;:7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª "ÿÿÿÿÿÿÿÿÿ# Ñ
G__inference_concatenate_layer_call_and_return_conditional_losses_142200[¢X
Q¢N
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿà
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿè
 ¨
,__inference_concatenate_layer_call_fn_142206x[¢X
Q¢N
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿà
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿè®
D__inference_conv1d_1_layer_call_and_return_conditional_losses_141484f*+4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÞ 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÞ 
 
)__inference_conv1d_1_layer_call_fn_141493Y*+4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÞ 
ª "ÿÿÿÿÿÿÿÿÿÞ ¬
B__inference_conv1d_layer_call_and_return_conditional_losses_141459f !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿR 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿR 
 
'__inference_conv1d_layer_call_fn_141468Y !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿR 
ª "ÿÿÿÿÿÿÿÿÿR ­
C__inference_dense_2_layer_call_and_return_conditional_losses_142490f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ#@
 
(__inference_dense_2_layer_call_fn_142499Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª "ÿÿÿÿÿÿÿÿÿ#@­
C__inference_dense_3_layer_call_and_return_conditional_losses_142529f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ#@
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 
(__inference_dense_3_layer_call_fn_142538Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ#@
ª "ÿÿÿÿÿÿÿÿÿ# ¤
C__inference_dense_4_layer_call_and_return_conditional_losses_142217]`a0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿè
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
(__inference_dense_4_layer_call_fn_142226P`a0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿè
ª "ÿÿÿÿÿÿÿÿÿ@£
C__inference_dense_5_layer_call_and_return_conditional_losses_142264\jk/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
(__inference_dense_5_layer_call_fn_142273Ojk/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@£
C__inference_dense_6_layer_call_and_return_conditional_losses_142310\tu/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_6_layer_call_fn_142319Otu/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dropout_4_layer_call_and_return_conditional_losses_142238\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¥
E__inference_dropout_4_layer_call_and_return_conditional_losses_142243\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
*__inference_dropout_4_layer_call_fn_142248O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@}
*__inference_dropout_4_layer_call_fn_142253O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@¥
E__inference_dropout_5_layer_call_and_return_conditional_losses_142285\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¥
E__inference_dropout_5_layer_call_and_return_conditional_losses_142290\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
*__inference_dropout_5_layer_call_fn_142295O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@}
*__inference_dropout_5_layer_call_fn_142300O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@¤
C__inference_flatten_layer_call_and_return_conditional_losses_142188]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿà
 |
(__inference_flatten_layer_call_fn_142193P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ# 
ª "ÿÿÿÿÿÿÿÿÿà
A__inference_model_layer_call_and_return_conditional_losses_140176À4~ !*+;<9:DEBC`ajktua¢^
W¢T
JG
"
input_1ÿÿÿÿÿÿÿÿÿR
!
input_2ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
A__inference_model_layer_call_and_return_conditional_losses_140270À4~ !*+<9;:EBDC`ajktua¢^
W¢T
JG
"
input_1ÿÿÿÿÿÿÿÿÿR
!
input_2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
A__inference_model_layer_call_and_return_conditional_losses_141011Â4~ !*+;<9:DEBC`ajktuc¢`
Y¢V
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿR
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
A__inference_model_layer_call_and_return_conditional_losses_141254Â4~ !*+<9;:EBDC`ajktuc¢`
Y¢V
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿR
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Þ
&__inference_model_layer_call_fn_140443³4~ !*+;<9:DEBC`ajktua¢^
W¢T
JG
"
input_1ÿÿÿÿÿÿÿÿÿR
!
input_2ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÞ
&__inference_model_layer_call_fn_140615³4~ !*+<9;:EBDC`ajktua¢^
W¢T
JG
"
input_1ÿÿÿÿÿÿÿÿÿR
!
input_2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿà
&__inference_model_layer_call_fn_141332µ4~ !*+;<9:DEBC`ajktuc¢`
Y¢V
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿR
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿà
&__inference_model_layer_call_fn_141410µ4~ !*+<9;:EBDC`ajktuc¢`
Y¢V
LI
# 
inputs/0ÿÿÿÿÿÿÿÿÿR
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÅ
H__inference_sequential_1_layer_call_and_return_conditional_losses_139243yB¢?
8¢5
+(
dense_2_inputÿÿÿÿÿÿÿÿÿ# 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 Å
H__inference_sequential_1_layer_call_and_return_conditional_losses_139257yB¢?
8¢5
+(
dense_2_inputÿÿÿÿÿÿÿÿÿ# 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ¾
H__inference_sequential_1_layer_call_and_return_conditional_losses_142376r;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ¾
H__inference_sequential_1_layer_call_and_return_conditional_losses_142433r;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 
-__inference_sequential_1_layer_call_fn_139285lB¢?
8¢5
+(
dense_2_inputÿÿÿÿÿÿÿÿÿ# 
p

 
ª "ÿÿÿÿÿÿÿÿÿ# 
-__inference_sequential_1_layer_call_fn_139312lB¢?
8¢5
+(
dense_2_inputÿÿÿÿÿÿÿÿÿ# 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ# 
-__inference_sequential_1_layer_call_fn_142446e;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p

 
ª "ÿÿÿÿÿÿÿÿÿ# 
-__inference_sequential_1_layer_call_fn_142459e;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ# þ
$__inference_signature_wrapper_140701Õ4~ !*+<9;:EBDC`ajktuj¢g
¢ 
`ª]
-
input_1"
input_1ÿÿÿÿÿÿÿÿÿR
,
input_2!
input_2ÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_6!
dense_6ÿÿÿÿÿÿÿÿÿ¹
X__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_141434]~+¢(
!¢

xÿÿÿÿÿÿÿÿÿR
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿR 
 
=__inference_token_and_position_embedding_layer_call_fn_141443P~+¢(
!¢

xÿÿÿÿÿÿÿÿÿR
ª "ÿÿÿÿÿÿÿÿÿR Ú
O__inference_transformer_block_1_layer_call_and_return_conditional_losses_141981 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 Ú
O__inference_transformer_block_1_layer_call_and_return_conditional_losses_142108 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ# 
 ±
4__inference_transformer_block_1_layer_call_fn_142145y 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p
ª "ÿÿÿÿÿÿÿÿÿ# ±
4__inference_transformer_block_1_layer_call_fn_142182y 7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ# 
p 
ª "ÿÿÿÿÿÿÿÿÿ# 