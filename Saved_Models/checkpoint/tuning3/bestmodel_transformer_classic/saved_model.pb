ЎЈ"
°ч
B
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
Љ
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
≠
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
Н
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
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
Н
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
dtypetypeИ
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
•
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
list(type)(0И
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

2	Р
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Њ
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
executor_typestring И
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
ц
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.12v2.4.1-0-g85c8b2a817f8ов
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
Њ
1token_and_position_embedding/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *B
shared_name31token_and_position_embedding/embedding/embeddings
Ј
Etoken_and_position_embedding/embedding/embeddings/Read/ReadVariableOpReadVariableOp1token_and_position_embedding/embedding/embeddings*
_output_shapes

: *
dtype0
√
3token_and_position_embedding/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Љi *D
shared_name53token_and_position_embedding/embedding_1/embeddings
Љ
Gtoken_and_position_embedding/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp3token_and_position_embedding/embedding_1/embeddings*
_output_shapes
:	Љi *
dtype0
ќ
7transformer_block_1/multi_head_attention_1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_1/multi_head_attention_1/query/kernel
«
Ktransformer_block_1/multi_head_attention_1/query/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_1/multi_head_attention_1/query/kernel*"
_output_shapes
:  *
dtype0
∆
5transformer_block_1/multi_head_attention_1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_1/multi_head_attention_1/query/bias
њ
Itransformer_block_1/multi_head_attention_1/query/bias/Read/ReadVariableOpReadVariableOp5transformer_block_1/multi_head_attention_1/query/bias*
_output_shapes

: *
dtype0
 
5transformer_block_1/multi_head_attention_1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *F
shared_name75transformer_block_1/multi_head_attention_1/key/kernel
√
Itransformer_block_1/multi_head_attention_1/key/kernel/Read/ReadVariableOpReadVariableOp5transformer_block_1/multi_head_attention_1/key/kernel*"
_output_shapes
:  *
dtype0
¬
3transformer_block_1/multi_head_attention_1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *D
shared_name53transformer_block_1/multi_head_attention_1/key/bias
ї
Gtransformer_block_1/multi_head_attention_1/key/bias/Read/ReadVariableOpReadVariableOp3transformer_block_1/multi_head_attention_1/key/bias*
_output_shapes

: *
dtype0
ќ
7transformer_block_1/multi_head_attention_1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_1/multi_head_attention_1/value/kernel
«
Ktransformer_block_1/multi_head_attention_1/value/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_1/multi_head_attention_1/value/kernel*"
_output_shapes
:  *
dtype0
∆
5transformer_block_1/multi_head_attention_1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_1/multi_head_attention_1/value/bias
њ
Itransformer_block_1/multi_head_attention_1/value/bias/Read/ReadVariableOpReadVariableOp5transformer_block_1/multi_head_attention_1/value/bias*
_output_shapes

: *
dtype0
д
Btransformer_block_1/multi_head_attention_1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBtransformer_block_1/multi_head_attention_1/attention_output/kernel
Ё
Vtransformer_block_1/multi_head_attention_1/attention_output/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_1/multi_head_attention_1/attention_output/kernel*"
_output_shapes
:  *
dtype0
Ў
@transformer_block_1/multi_head_attention_1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_1/multi_head_attention_1/attention_output/bias
—
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
ґ
/transformer_block_1/layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_1/layer_normalization_2/gamma
ѓ
Ctransformer_block_1/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_1/layer_normalization_2/gamma*
_output_shapes
: *
dtype0
і
.transformer_block_1/layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_1/layer_normalization_2/beta
≠
Btransformer_block_1/layer_normalization_2/beta/Read/ReadVariableOpReadVariableOp.transformer_block_1/layer_normalization_2/beta*
_output_shapes
: *
dtype0
ґ
/transformer_block_1/layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_1/layer_normalization_3/gamma
ѓ
Ctransformer_block_1/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_1/layer_normalization_3/gamma*
_output_shapes
: *
dtype0
і
.transformer_block_1/layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_1/layer_normalization_3/beta
≠
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
Т
SGD/dense_4/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *,
shared_nameSGD/dense_4/kernel/momentum
Л
/SGD/dense_4/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_4/kernel/momentum*
_output_shapes

:( *
dtype0
К
SGD/dense_4/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameSGD/dense_4/bias/momentum
Г
-SGD/dense_4/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_4/bias/momentum*
_output_shapes
: *
dtype0
Т
SGD/dense_5/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *,
shared_nameSGD/dense_5/kernel/momentum
Л
/SGD/dense_5/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_5/kernel/momentum*
_output_shapes

:  *
dtype0
К
SGD/dense_5/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameSGD/dense_5/bias/momentum
Г
-SGD/dense_5/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_5/bias/momentum*
_output_shapes
: *
dtype0
Т
SGD/dense_6/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *,
shared_nameSGD/dense_6/kernel/momentum
Л
/SGD/dense_6/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_6/kernel/momentum*
_output_shapes

: *
dtype0
К
SGD/dense_6/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameSGD/dense_6/bias/momentum
Г
-SGD/dense_6/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_6/bias/momentum*
_output_shapes
:*
dtype0
Ў
>SGD/token_and_position_embedding/embedding/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *O
shared_name@>SGD/token_and_position_embedding/embedding/embeddings/momentum
—
RSGD/token_and_position_embedding/embedding/embeddings/momentum/Read/ReadVariableOpReadVariableOp>SGD/token_and_position_embedding/embedding/embeddings/momentum*
_output_shapes

: *
dtype0
Ё
@SGD/token_and_position_embedding/embedding_1/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Љi *Q
shared_nameB@SGD/token_and_position_embedding/embedding_1/embeddings/momentum
÷
TSGD/token_and_position_embedding/embedding_1/embeddings/momentum/Read/ReadVariableOpReadVariableOp@SGD/token_and_position_embedding/embedding_1/embeddings/momentum*
_output_shapes
:	Љi *
dtype0
и
DSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentum
б
XSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentum*"
_output_shapes
:  *
dtype0
а
BSGD/transformer_block_1/multi_head_attention_1/query/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_1/multi_head_attention_1/query/bias/momentum
ў
VSGD/transformer_block_1/multi_head_attention_1/query/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_1/multi_head_attention_1/query/bias/momentum*
_output_shapes

: *
dtype0
д
BSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentum
Ё
VSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentum*"
_output_shapes
:  *
dtype0
№
@SGD/transformer_block_1/multi_head_attention_1/key/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *Q
shared_nameB@SGD/transformer_block_1/multi_head_attention_1/key/bias/momentum
’
TSGD/transformer_block_1/multi_head_attention_1/key/bias/momentum/Read/ReadVariableOpReadVariableOp@SGD/transformer_block_1/multi_head_attention_1/key/bias/momentum*
_output_shapes

: *
dtype0
и
DSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentum
б
XSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentum*"
_output_shapes
:  *
dtype0
а
BSGD/transformer_block_1/multi_head_attention_1/value/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_1/multi_head_attention_1/value/bias/momentum
ў
VSGD/transformer_block_1/multi_head_attention_1/value/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_1/multi_head_attention_1/value/bias/momentum*
_output_shapes

: *
dtype0
ю
OSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *`
shared_nameQOSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentum
ч
cSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentum/Read/ReadVariableOpReadVariableOpOSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentum*"
_output_shapes
:  *
dtype0
т
MSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *^
shared_nameOMSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentum
л
aSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentum/Read/ReadVariableOpReadVariableOpMSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentum*
_output_shapes
: *
dtype0
Т
SGD/dense_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *,
shared_nameSGD/dense_2/kernel/momentum
Л
/SGD/dense_2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_2/kernel/momentum*
_output_shapes

:  *
dtype0
К
SGD/dense_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameSGD/dense_2/bias/momentum
Г
-SGD/dense_2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_2/bias/momentum*
_output_shapes
: *
dtype0
Т
SGD/dense_3/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *,
shared_nameSGD/dense_3/kernel/momentum
Л
/SGD/dense_3/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_3/kernel/momentum*
_output_shapes

:  *
dtype0
К
SGD/dense_3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameSGD/dense_3/bias/momentum
Г
-SGD/dense_3/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_3/bias/momentum*
_output_shapes
: *
dtype0
–
<SGD/transformer_block_1/layer_normalization_2/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_1/layer_normalization_2/gamma/momentum
…
PSGD/transformer_block_1/layer_normalization_2/gamma/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_1/layer_normalization_2/gamma/momentum*
_output_shapes
: *
dtype0
ќ
;SGD/transformer_block_1/layer_normalization_2/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;SGD/transformer_block_1/layer_normalization_2/beta/momentum
«
OSGD/transformer_block_1/layer_normalization_2/beta/momentum/Read/ReadVariableOpReadVariableOp;SGD/transformer_block_1/layer_normalization_2/beta/momentum*
_output_shapes
: *
dtype0
–
<SGD/transformer_block_1/layer_normalization_3/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_1/layer_normalization_3/gamma/momentum
…
PSGD/transformer_block_1/layer_normalization_3/gamma/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_1/layer_normalization_3/gamma/momentum*
_output_shapes
: *
dtype0
ќ
;SGD/transformer_block_1/layer_normalization_3/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;SGD/transformer_block_1/layer_normalization_3/beta/momentum
«
OSGD/transformer_block_1/layer_normalization_3/beta/momentum/Read/ReadVariableOpReadVariableOp;SGD/transformer_block_1/layer_normalization_3/beta/momentum*
_output_shapes
: *
dtype0

NoOpNoOp
ЋК
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЕК
valueъЙBцЙ BоЙ
С
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
†
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
ќ
	Idecay
Jlearning_rate
Kmomentum
Liter/momentumљ0momentumЊ9momentumњ:momentumјCmomentumЅDmomentum¬Mmomentum√NmomentumƒOmomentum≈Pmomentum∆Qmomentum«Rmomentum»Smomentum…Tmomentum UmomentumЋVmomentumћWmomentumЌXmomentumќYmomentumѕZmomentum–[momentum—\momentum“]momentum”^momentum‘
ґ
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
ґ
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
≠
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
≠
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
≠
qnon_trainable_variables
rmetrics
slayer_metrics
trainable_variables
tlayer_regularization_losses

ulayers
	variables
regularization_losses
ї
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
®
Аlayer_with_weights-0
Аlayer-0
Бlayer_with_weights-1
Бlayer-1
Вtrainable_variables
Г	variables
Дregularization_losses
Е	keras_api
v
	Жaxis
	[gamma
\beta
Зtrainable_variables
И	variables
Йregularization_losses
К	keras_api
v
	Лaxis
	]gamma
^beta
Мtrainable_variables
Н	variables
Оregularization_losses
П	keras_api
V
Рtrainable_variables
С	variables
Тregularization_losses
У	keras_api
V
Фtrainable_variables
Х	variables
Цregularization_losses
Ч	keras_api
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
≤
Шnon_trainable_variables
Щmetrics
Ъlayer_metrics
#trainable_variables
 Ыlayer_regularization_losses
Ьlayers
$	variables
%regularization_losses
 
 
 
≤
Эnon_trainable_variables
Юmetrics
Яlayer_metrics
'trainable_variables
 †layer_regularization_losses
°layers
(	variables
)regularization_losses
 
 
 
≤
Ґnon_trainable_variables
£metrics
§layer_metrics
+trainable_variables
 •layer_regularization_losses
¶layers
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
≤
Іnon_trainable_variables
®metrics
©layer_metrics
1trainable_variables
 ™layer_regularization_losses
Ђlayers
2	variables
3regularization_losses
 
 
 
≤
ђnon_trainable_variables
≠metrics
Ѓlayer_metrics
5trainable_variables
 ѓlayer_regularization_losses
∞layers
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
≤
±non_trainable_variables
≤metrics
≥layer_metrics
;trainable_variables
 іlayer_regularization_losses
µlayers
<	variables
=regularization_losses
 
 
 
≤
ґnon_trainable_variables
Јmetrics
Єlayer_metrics
?trainable_variables
 єlayer_regularization_losses
Їlayers
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
≤
їnon_trainable_variables
Љmetrics
љlayer_metrics
Etrainable_variables
 Њlayer_regularization_losses
њlayers
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
ЙЖ
VARIABLE_VALUEBtransformer_block_1/multi_head_attention_1/attention_output/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
ЗД
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
ј0
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
≤
Ѕnon_trainable_variables
¬metrics
√layer_metrics
dtrainable_variables
 ƒlayer_regularization_losses
≈layers
e	variables
fregularization_losses

N0

N0
 
≤
∆non_trainable_variables
«metrics
»layer_metrics
htrainable_variables
 …layer_regularization_losses
 layers
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
Я
Ћpartial_output_shape
ћfull_output_shape

Okernel
Pbias
Ќtrainable_variables
ќ	variables
ѕregularization_losses
–	keras_api
Я
—partial_output_shape
“full_output_shape

Qkernel
Rbias
”trainable_variables
‘	variables
’regularization_losses
÷	keras_api
Я
„partial_output_shape
Ўfull_output_shape

Skernel
Tbias
ўtrainable_variables
Џ	variables
џregularization_losses
№	keras_api
V
Ёtrainable_variables
ё	variables
яregularization_losses
а	keras_api
V
бtrainable_variables
в	variables
гregularization_losses
д	keras_api
Я
еpartial_output_shape
жfull_output_shape

Ukernel
Vbias
зtrainable_variables
и	variables
йregularization_losses
к	keras_api
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
≤
лnon_trainable_variables
мmetrics
нlayer_metrics
|trainable_variables
 оlayer_regularization_losses
пlayers
}	variables
~regularization_losses
l

Wkernel
Xbias
рtrainable_variables
с	variables
тregularization_losses
у	keras_api
l

Ykernel
Zbias
фtrainable_variables
х	variables
цregularization_losses
ч	keras_api

W0
X1
Y2
Z3

W0
X1
Y2
Z3
 
µ
шnon_trainable_variables
щmetrics
ъlayer_metrics
Вtrainable_variables
 ыlayer_regularization_losses
ьlayers
Г	variables
Дregularization_losses
 

[0
\1

[0
\1
 
µ
эnon_trainable_variables
юmetrics
€layer_metrics
Зtrainable_variables
 Аlayer_regularization_losses
Бlayers
И	variables
Йregularization_losses
 

]0
^1

]0
^1
 
µ
Вnon_trainable_variables
Гmetrics
Дlayer_metrics
Мtrainable_variables
 Еlayer_regularization_losses
Жlayers
Н	variables
Оregularization_losses
 
 
 
µ
Зnon_trainable_variables
Иmetrics
Йlayer_metrics
Рtrainable_variables
 Кlayer_regularization_losses
Лlayers
С	variables
Тregularization_losses
 
 
 
µ
Мnon_trainable_variables
Нmetrics
Оlayer_metrics
Фtrainable_variables
 Пlayer_regularization_losses
Рlayers
Х	variables
Цregularization_losses
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

Сtotal

Тcount
У	variables
Ф	keras_api
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
µ
Хnon_trainable_variables
Цmetrics
Чlayer_metrics
Ќtrainable_variables
 Шlayer_regularization_losses
Щlayers
ќ	variables
ѕregularization_losses
 
 

Q0
R1

Q0
R1
 
µ
Ъnon_trainable_variables
Ыmetrics
Ьlayer_metrics
”trainable_variables
 Эlayer_regularization_losses
Юlayers
‘	variables
’regularization_losses
 
 

S0
T1

S0
T1
 
µ
Яnon_trainable_variables
†metrics
°layer_metrics
ўtrainable_variables
 Ґlayer_regularization_losses
£layers
Џ	variables
џregularization_losses
 
 
 
µ
§non_trainable_variables
•metrics
¶layer_metrics
Ёtrainable_variables
 Іlayer_regularization_losses
®layers
ё	variables
яregularization_losses
 
 
 
µ
©non_trainable_variables
™metrics
Ђlayer_metrics
бtrainable_variables
 ђlayer_regularization_losses
≠layers
в	variables
гregularization_losses
 
 

U0
V1

U0
V1
 
µ
Ѓnon_trainable_variables
ѓmetrics
∞layer_metrics
зtrainable_variables
 ±layer_regularization_losses
≤layers
и	variables
йregularization_losses
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
µ
≥non_trainable_variables
іmetrics
µlayer_metrics
рtrainable_variables
 ґlayer_regularization_losses
Јlayers
с	variables
тregularization_losses

Y0
Z1

Y0
Z1
 
µ
Єnon_trainable_variables
єmetrics
Їlayer_metrics
фtrainable_variables
 їlayer_regularization_losses
Љlayers
х	variables
цregularization_losses
 
 
 
 

А0
Б1
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
С0
Т1

У	variables
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
ЛИ
VARIABLE_VALUESGD/dense_4/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUESGD/dense_4/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUESGD/dense_5/kernel/momentumYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUESGD/dense_5/bias/momentumWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUESGD/dense_6/kernel/momentumYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUESGD/dense_6/bias/momentumWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
®•
VARIABLE_VALUE>SGD/token_and_position_embedding/embedding/embeddings/momentumStrainable_variables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
™І
VARIABLE_VALUE@SGD/token_and_position_embedding/embedding_1/embeddings/momentumStrainable_variables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЃЂ
VARIABLE_VALUEDSGD/transformer_block_1/multi_head_attention_1/query/kernel/momentumStrainable_variables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ђ©
VARIABLE_VALUEBSGD/transformer_block_1/multi_head_attention_1/query/bias/momentumStrainable_variables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ђ©
VARIABLE_VALUEBSGD/transformer_block_1/multi_head_attention_1/key/kernel/momentumStrainable_variables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
™І
VARIABLE_VALUE@SGD/transformer_block_1/multi_head_attention_1/key/bias/momentumStrainable_variables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЃЂ
VARIABLE_VALUEDSGD/transformer_block_1/multi_head_attention_1/value/kernel/momentumStrainable_variables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ђ©
VARIABLE_VALUEBSGD/transformer_block_1/multi_head_attention_1/value/bias/momentumStrainable_variables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
єґ
VARIABLE_VALUEOSGD/transformer_block_1/multi_head_attention_1/attention_output/kernel/momentumStrainable_variables/8/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
Јі
VARIABLE_VALUEMSGD/transformer_block_1/multi_head_attention_1/attention_output/bias/momentumStrainable_variables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUESGD/dense_2/kernel/momentumTtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUESGD/dense_2/bias/momentumTtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUESGD/dense_3/kernel/momentumTtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUESGD/dense_3/bias/momentumTtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
І§
VARIABLE_VALUE<SGD/transformer_block_1/layer_normalization_2/gamma/momentumTtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¶£
VARIABLE_VALUE;SGD/transformer_block_1/layer_normalization_2/beta/momentumTtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
І§
VARIABLE_VALUE<SGD/transformer_block_1/layer_normalization_3/gamma/momentumTtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
¶£
VARIABLE_VALUE;SGD/transformer_block_1/layer_normalization_3/beta/momentumTtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:€€€€€€€€€Љi*
dtype0*
shape:€€€€€€€€€Љi
z
serving_default_input_2Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Р

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_23token_and_position_embedding/embedding_1/embeddings1token_and_position_embedding/embedding/embeddings7transformer_block_1/multi_head_attention_1/query/kernel5transformer_block_1/multi_head_attention_1/query/bias5transformer_block_1/multi_head_attention_1/key/kernel3transformer_block_1/multi_head_attention_1/key/bias7transformer_block_1/multi_head_attention_1/value/kernel5transformer_block_1/multi_head_attention_1/value/biasBtransformer_block_1/multi_head_attention_1/attention_output/kernel@transformer_block_1/multi_head_attention_1/attention_output/bias/transformer_block_1/layer_normalization_2/gamma.transformer_block_1/layer_normalization_2/betadense_2/kerneldense_2/biasdense_3/kerneldense_3/bias/transformer_block_1/layer_normalization_3/gamma.transformer_block_1/layer_normalization_3/betadense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/bias*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference_signature_wrapper_89447
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Я
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
GPU2*0J 8В *'
f"R 
__inference__traced_save_90881
в
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
GPU2*0J 8В **
f%R#
!__inference__traced_restore_91053√О
ї
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
concat/axisБ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€(2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€ :€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
Лр
Я
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
identityИҐdense_4/BiasAdd/ReadVariableOpҐdense_4/MatMul/ReadVariableOpҐdense_5/BiasAdd/ReadVariableOpҐdense_5/MatMul/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐ7token_and_position_embedding/embedding/embedding_lookupҐ9token_and_position_embedding/embedding_1/embedding_lookupҐBtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpҐFtransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpҐBtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpҐFtransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpҐNtransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpҐXtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpҐAtransformer_block_1/multi_head_attention_1/key/add/ReadVariableOpҐKtransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpҐCtransformer_block_1/multi_head_attention_1/query/add/ReadVariableOpҐMtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpҐCtransformer_block_1/multi_head_attention_1/value/add/ReadVariableOpҐMtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpҐ?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpҐAtransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpҐ?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpҐAtransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpА
"token_and_position_embedding/ShapeShapeinputs_0*
T0*
_output_shapes
:2$
"token_and_position_embedding/ShapeЈ
0token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€22
0token_and_position_embedding/strided_slice/stack≤
2token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 24
2token_and_position_embedding/strided_slice/stack_1≤
2token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2token_and_position_embedding/strided_slice/stack_2Р
*token_and_position_embedding/strided_sliceStridedSlice+token_and_position_embedding/Shape:output:09token_and_position_embedding/strided_slice/stack:output:0;token_and_position_embedding/strided_slice/stack_1:output:0;token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*token_and_position_embedding/strided_sliceЦ
(token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2*
(token_and_position_embedding/range/startЦ
(token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2*
(token_and_position_embedding/range/deltaС
"token_and_position_embedding/rangeRange1token_and_position_embedding/range/start:output:03token_and_position_embedding/strided_slice:output:01token_and_position_embedding/range/delta:output:0*#
_output_shapes
:€€€€€€€€€2$
"token_and_position_embedding/rangeЊ
9token_and_position_embedding/embedding_1/embedding_lookupResourceGather?token_and_position_embedding_embedding_1_embedding_lookup_89672+token_and_position_embedding/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*R
_classH
FDloc:@token_and_position_embedding/embedding_1/embedding_lookup/89672*'
_output_shapes
:€€€€€€€€€ *
dtype02;
9token_and_position_embedding/embedding_1/embedding_lookupМ
Btoken_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityBtoken_and_position_embedding/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*R
_classH
FDloc:@token_and_position_embedding/embedding_1/embedding_lookup/89672*'
_output_shapes
:€€€€€€€€€ 2D
Btoken_and_position_embedding/embedding_1/embedding_lookup/IdentityЧ
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityKtoken_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2F
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1Ѓ
+token_and_position_embedding/embedding/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€Љi2-
+token_and_position_embedding/embedding/Castњ
7token_and_position_embedding/embedding/embedding_lookupResourceGather=token_and_position_embedding_embedding_embedding_lookup_89678/token_and_position_embedding/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@token_and_position_embedding/embedding/embedding_lookup/89678*,
_output_shapes
:€€€€€€€€€Љi *
dtype029
7token_and_position_embedding/embedding/embedding_lookupЙ
@token_and_position_embedding/embedding/embedding_lookup/IdentityIdentity@token_and_position_embedding/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@token_and_position_embedding/embedding/embedding_lookup/89678*,
_output_shapes
:€€€€€€€€€Љi 2B
@token_and_position_embedding/embedding/embedding_lookup/IdentityЦ
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityItoken_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€Љi 2D
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1†
 token_and_position_embedding/addAddV2Ktoken_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Mtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Љi 2"
 token_and_position_embedding/addЖ
 average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 average_pooling1d/ExpandDims/dim÷
average_pooling1d/ExpandDims
ExpandDims$token_and_position_embedding/add:z:0)average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Љi 2
average_pooling1d/ExpandDimsа
average_pooling1d/AvgPoolAvgPool%average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:€€€€€€€€€- *
ksize	
ђ*
paddingVALID*
strides	
ђ2
average_pooling1d/AvgPool≤
average_pooling1d/SqueezeSqueeze"average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€- *
squeeze_dims
2
average_pooling1d/Squeezeє
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_1_multi_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpе
>transformer_block_1/multi_head_attention_1/query/einsum/EinsumEinsum"average_pooling1d/Squeeze:output:0Utransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2@
>transformer_block_1/multi_head_attention_1/query/einsum/EinsumЧ
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOpReadVariableOpLtransformer_block_1_multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp≈
4transformer_block_1/multi_head_attention_1/query/addAddV2Gtransformer_block_1/multi_head_attention_1/query/einsum/Einsum:output:0Ktransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 26
4transformer_block_1/multi_head_attention_1/query/add≥
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_1_multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpя
<transformer_block_1/multi_head_attention_1/key/einsum/EinsumEinsum"average_pooling1d/Squeeze:output:0Stransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2>
<transformer_block_1/multi_head_attention_1/key/einsum/EinsumС
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOpReadVariableOpJtransformer_block_1_multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOpљ
2transformer_block_1/multi_head_attention_1/key/addAddV2Etransformer_block_1/multi_head_attention_1/key/einsum/Einsum:output:0Itransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 24
2transformer_block_1/multi_head_attention_1/key/addє
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_1_multi_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpе
>transformer_block_1/multi_head_attention_1/value/einsum/EinsumEinsum"average_pooling1d/Squeeze:output:0Utransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2@
>transformer_block_1/multi_head_attention_1/value/einsum/EinsumЧ
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOpReadVariableOpLtransformer_block_1_multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp≈
4transformer_block_1/multi_head_attention_1/value/addAddV2Gtransformer_block_1/multi_head_attention_1/value/einsum/Einsum:output:0Ktransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 26
4transformer_block_1/multi_head_attention_1/value/add©
0transformer_block_1/multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>22
0transformer_block_1/multi_head_attention_1/Mul/yЦ
.transformer_block_1/multi_head_attention_1/MulMul8transformer_block_1/multi_head_attention_1/query/add:z:09transformer_block_1/multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€- 20
.transformer_block_1/multi_head_attention_1/Mulћ
8transformer_block_1/multi_head_attention_1/einsum/EinsumEinsum6transformer_block_1/multi_head_attention_1/key/add:z:02transformer_block_1/multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€--*
equationaecd,abcd->acbe2:
8transformer_block_1/multi_head_attention_1/einsum/EinsumА
:transformer_block_1/multi_head_attention_1/softmax/SoftmaxSoftmaxAtransformer_block_1/multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€--2<
:transformer_block_1/multi_head_attention_1/softmax/SoftmaxЖ
;transformer_block_1/multi_head_attention_1/dropout/IdentityIdentityDtransformer_block_1/multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€--2=
;transformer_block_1/multi_head_attention_1/dropout/Identityд
:transformer_block_1/multi_head_attention_1/einsum_1/EinsumEinsumDtransformer_block_1/multi_head_attention_1/dropout/Identity:output:08transformer_block_1/multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationacbe,aecd->abcd2<
:transformer_block_1/multi_head_attention_1/einsum_1/EinsumЏ
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_1_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp£
Itransformer_block_1/multi_head_attention_1/attention_output/einsum/EinsumEinsumCtransformer_block_1/multi_head_attention_1/einsum_1/Einsum:output:0`transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€- *
equationabcd,cde->abe2K
Itransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsumі
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_1_multi_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpн
?transformer_block_1/multi_head_attention_1/attention_output/addAddV2Rtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum:output:0Vtransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2A
?transformer_block_1/multi_head_attention_1/attention_output/add„
&transformer_block_1/dropout_2/IdentityIdentityCtransformer_block_1/multi_head_attention_1/attention_output/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2(
&transformer_block_1/dropout_2/Identity∆
transformer_block_1/addAddV2"average_pooling1d/Squeeze:output:0/transformer_block_1/dropout_2/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
transformer_block_1/addё
Htransformer_block_1/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_1/layer_normalization_2/moments/mean/reduction_indicesѓ
6transformer_block_1/layer_normalization_2/moments/meanMeantransformer_block_1/add:z:0Qtransformer_block_1/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(28
6transformer_block_1/layer_normalization_2/moments/meanЗ
>transformer_block_1/layer_normalization_2/moments/StopGradientStopGradient?transformer_block_1/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2@
>transformer_block_1/layer_normalization_2/moments/StopGradientї
Ctransformer_block_1/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencetransformer_block_1/add:z:0Gtransformer_block_1/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2E
Ctransformer_block_1/layer_normalization_2/moments/SquaredDifferenceж
Ltransformer_block_1/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_1/layer_normalization_2/moments/variance/reduction_indicesз
:transformer_block_1/layer_normalization_2/moments/varianceMeanGtransformer_block_1/layer_normalization_2/moments/SquaredDifference:z:0Utransformer_block_1/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2<
:transformer_block_1/layer_normalization_2/moments/varianceї
9transformer_block_1/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52;
9transformer_block_1/layer_normalization_2/batchnorm/add/yЇ
7transformer_block_1/layer_normalization_2/batchnorm/addAddV2Ctransformer_block_1/layer_normalization_2/moments/variance:output:0Btransformer_block_1/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€-29
7transformer_block_1/layer_normalization_2/batchnorm/addт
9transformer_block_1/layer_normalization_2/batchnorm/RsqrtRsqrt;transformer_block_1/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€-2;
9transformer_block_1/layer_normalization_2/batchnorm/RsqrtЬ
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpЊ
7transformer_block_1/layer_normalization_2/batchnorm/mulMul=transformer_block_1/layer_normalization_2/batchnorm/Rsqrt:y:0Ntransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 29
7transformer_block_1/layer_normalization_2/batchnorm/mulН
9transformer_block_1/layer_normalization_2/batchnorm/mul_1Multransformer_block_1/add:z:0;transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2;
9transformer_block_1/layer_normalization_2/batchnorm/mul_1±
9transformer_block_1/layer_normalization_2/batchnorm/mul_2Mul?transformer_block_1/layer_normalization_2/moments/mean:output:0;transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2;
9transformer_block_1/layer_normalization_2/batchnorm/mul_2Р
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_1_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpЇ
7transformer_block_1/layer_normalization_2/batchnorm/subSubJtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp:value:0=transformer_block_1/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 29
7transformer_block_1/layer_normalization_2/batchnorm/sub±
9transformer_block_1/layer_normalization_2/batchnorm/add_1AddV2=transformer_block_1/layer_normalization_2/batchnorm/mul_1:z:0;transformer_block_1/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2;
9transformer_block_1/layer_normalization_2/batchnorm/add_1С
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_1_sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02C
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpЉ
7transformer_block_1/sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_1/sequential_1/dense_2/Tensordot/axes√
7transformer_block_1/sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_1/sequential_1/dense_2/Tensordot/freeб
8transformer_block_1/sequential_1/dense_2/Tensordot/ShapeShape=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_2/Tensordot/Shape∆
@transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axisЮ
;transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2GatherV2Atransformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2 
Btransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axis§
=transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1GatherV2Atransformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Ktransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1Њ
8transformer_block_1/sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_1/sequential_1/dense_2/Tensordot/Const§
7transformer_block_1/sequential_1/dense_2/Tensordot/ProdProdDtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Atransformer_block_1/sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_1/sequential_1/dense_2/Tensordot/Prod¬
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_1ђ
9transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1ProdFtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1:output:0Ctransformer_block_1/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1¬
>transformer_block_1/sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_1/sequential_1/dense_2/Tensordot/concat/axisэ
9transformer_block_1/sequential_1/dense_2/Tensordot/concatConcatV2@transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Gtransformer_block_1/sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_1/sequential_1/dense_2/Tensordot/concat∞
8transformer_block_1/sequential_1/dense_2/Tensordot/stackPack@transformer_block_1/sequential_1/dense_2/Tensordot/Prod:output:0Btransformer_block_1/sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_2/Tensordot/stack¬
<transformer_block_1/sequential_1/dense_2/Tensordot/transpose	Transpose=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0Btransformer_block_1/sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2>
<transformer_block_1/sequential_1/dense_2/Tensordot/transpose√
:transformer_block_1/sequential_1/dense_2/Tensordot/ReshapeReshape@transformer_block_1/sequential_1/dense_2/Tensordot/transpose:y:0Atransformer_block_1/sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2<
:transformer_block_1/sequential_1/dense_2/Tensordot/Reshape¬
9transformer_block_1/sequential_1/dense_2/Tensordot/MatMulMatMulCtransformer_block_1/sequential_1/dense_2/Tensordot/Reshape:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2;
9transformer_block_1/sequential_1/dense_2/Tensordot/MatMul¬
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_2∆
@transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axisК
;transformer_block_1/sequential_1/dense_2/Tensordot/concat_1ConcatV2Dtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Ctransformer_block_1/sequential_1/dense_2/Tensordot/Const_2:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_2/Tensordot/concat_1і
2transformer_block_1/sequential_1/dense_2/TensordotReshapeCtransformer_block_1/sequential_1/dense_2/Tensordot/MatMul:product:0Dtransformer_block_1/sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 24
2transformer_block_1/sequential_1/dense_2/TensordotЗ
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_1_sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpЂ
0transformer_block_1/sequential_1/dense_2/BiasAddBiasAdd;transformer_block_1/sequential_1/dense_2/Tensordot:output:0Gtransformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 22
0transformer_block_1/sequential_1/dense_2/BiasAdd„
-transformer_block_1/sequential_1/dense_2/ReluRelu9transformer_block_1/sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2/
-transformer_block_1/sequential_1/dense_2/ReluС
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_1_sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02C
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpЉ
7transformer_block_1/sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_1/sequential_1/dense_3/Tensordot/axes√
7transformer_block_1/sequential_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_1/sequential_1/dense_3/Tensordot/freeя
8transformer_block_1/sequential_1/dense_3/Tensordot/ShapeShape;transformer_block_1/sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_3/Tensordot/Shape∆
@transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axisЮ
;transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2GatherV2Atransformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2 
Btransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axis§
=transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1GatherV2Atransformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Ktransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1Њ
8transformer_block_1/sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_1/sequential_1/dense_3/Tensordot/Const§
7transformer_block_1/sequential_1/dense_3/Tensordot/ProdProdDtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Atransformer_block_1/sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_1/sequential_1/dense_3/Tensordot/Prod¬
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_1ђ
9transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1ProdFtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1:output:0Ctransformer_block_1/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1¬
>transformer_block_1/sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_1/sequential_1/dense_3/Tensordot/concat/axisэ
9transformer_block_1/sequential_1/dense_3/Tensordot/concatConcatV2@transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Gtransformer_block_1/sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_1/sequential_1/dense_3/Tensordot/concat∞
8transformer_block_1/sequential_1/dense_3/Tensordot/stackPack@transformer_block_1/sequential_1/dense_3/Tensordot/Prod:output:0Btransformer_block_1/sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_3/Tensordot/stackј
<transformer_block_1/sequential_1/dense_3/Tensordot/transpose	Transpose;transformer_block_1/sequential_1/dense_2/Relu:activations:0Btransformer_block_1/sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2>
<transformer_block_1/sequential_1/dense_3/Tensordot/transpose√
:transformer_block_1/sequential_1/dense_3/Tensordot/ReshapeReshape@transformer_block_1/sequential_1/dense_3/Tensordot/transpose:y:0Atransformer_block_1/sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2<
:transformer_block_1/sequential_1/dense_3/Tensordot/Reshape¬
9transformer_block_1/sequential_1/dense_3/Tensordot/MatMulMatMulCtransformer_block_1/sequential_1/dense_3/Tensordot/Reshape:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2;
9transformer_block_1/sequential_1/dense_3/Tensordot/MatMul¬
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_2∆
@transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axisК
;transformer_block_1/sequential_1/dense_3/Tensordot/concat_1ConcatV2Dtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Ctransformer_block_1/sequential_1/dense_3/Tensordot/Const_2:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_3/Tensordot/concat_1і
2transformer_block_1/sequential_1/dense_3/TensordotReshapeCtransformer_block_1/sequential_1/dense_3/Tensordot/MatMul:product:0Dtransformer_block_1/sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 24
2transformer_block_1/sequential_1/dense_3/TensordotЗ
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_1_sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpЂ
0transformer_block_1/sequential_1/dense_3/BiasAddBiasAdd;transformer_block_1/sequential_1/dense_3/Tensordot:output:0Gtransformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 22
0transformer_block_1/sequential_1/dense_3/BiasAddЌ
&transformer_block_1/dropout_3/IdentityIdentity9transformer_block_1/sequential_1/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2(
&transformer_block_1/dropout_3/Identityе
transformer_block_1/add_1AddV2=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0/transformer_block_1/dropout_3/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
transformer_block_1/add_1ё
Htransformer_block_1/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_1/layer_normalization_3/moments/mean/reduction_indices±
6transformer_block_1/layer_normalization_3/moments/meanMeantransformer_block_1/add_1:z:0Qtransformer_block_1/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(28
6transformer_block_1/layer_normalization_3/moments/meanЗ
>transformer_block_1/layer_normalization_3/moments/StopGradientStopGradient?transformer_block_1/layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2@
>transformer_block_1/layer_normalization_3/moments/StopGradientљ
Ctransformer_block_1/layer_normalization_3/moments/SquaredDifferenceSquaredDifferencetransformer_block_1/add_1:z:0Gtransformer_block_1/layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2E
Ctransformer_block_1/layer_normalization_3/moments/SquaredDifferenceж
Ltransformer_block_1/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_1/layer_normalization_3/moments/variance/reduction_indicesз
:transformer_block_1/layer_normalization_3/moments/varianceMeanGtransformer_block_1/layer_normalization_3/moments/SquaredDifference:z:0Utransformer_block_1/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2<
:transformer_block_1/layer_normalization_3/moments/varianceї
9transformer_block_1/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52;
9transformer_block_1/layer_normalization_3/batchnorm/add/yЇ
7transformer_block_1/layer_normalization_3/batchnorm/addAddV2Ctransformer_block_1/layer_normalization_3/moments/variance:output:0Btransformer_block_1/layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€-29
7transformer_block_1/layer_normalization_3/batchnorm/addт
9transformer_block_1/layer_normalization_3/batchnorm/RsqrtRsqrt;transformer_block_1/layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€-2;
9transformer_block_1/layer_normalization_3/batchnorm/RsqrtЬ
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpЊ
7transformer_block_1/layer_normalization_3/batchnorm/mulMul=transformer_block_1/layer_normalization_3/batchnorm/Rsqrt:y:0Ntransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 29
7transformer_block_1/layer_normalization_3/batchnorm/mulП
9transformer_block_1/layer_normalization_3/batchnorm/mul_1Multransformer_block_1/add_1:z:0;transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2;
9transformer_block_1/layer_normalization_3/batchnorm/mul_1±
9transformer_block_1/layer_normalization_3/batchnorm/mul_2Mul?transformer_block_1/layer_normalization_3/moments/mean:output:0;transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2;
9transformer_block_1/layer_normalization_3/batchnorm/mul_2Р
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpЇ
7transformer_block_1/layer_normalization_3/batchnorm/subSubJtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp:value:0=transformer_block_1/layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 29
7transformer_block_1/layer_normalization_3/batchnorm/sub±
9transformer_block_1/layer_normalization_3/batchnorm/add_1AddV2=transformer_block_1/layer_normalization_3/batchnorm/mul_1:z:0;transformer_block_1/layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2;
9transformer_block_1/layer_normalization_3/batchnorm/add_1§
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indicesс
global_average_pooling1d/MeanMean=transformer_block_1/layer_normalization_3/batchnorm/add_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
global_average_pooling1d/Meant
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis√
concatenate/concatConcatV2&global_average_pooling1d/Mean:output:0inputs_1 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€(2
concatenate/concat•
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:( *
dtype02
dense_4/MatMul/ReadVariableOp†
dense_4/MatMulMatMulconcatenate/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_4/MatMul§
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOp°
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_4/ReluВ
dropout_4/IdentityIdentitydense_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_4/Identity•
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_5/MatMul/ReadVariableOp†
dense_5/MatMulMatMuldropout_4/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_5/MatMul§
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_5/BiasAdd/ReadVariableOp°
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_5/ReluВ
dropout_5/IdentityIdentitydense_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_5/Identity•
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_6/MatMul/ReadVariableOp†
dense_6/MatMulMatMuldropout_5/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_6/MatMul§
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp°
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_6/BiasAddЈ
IdentityIdentitydense_6/BiasAdd:output:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp8^token_and_position_embedding/embedding/embedding_lookup:^token_and_position_embedding/embedding_1/embedding_lookupC^transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpG^transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpC^transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpG^transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpO^transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpY^transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_1/multi_head_attention_1/key/add/ReadVariableOpL^transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpD^transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpN^transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpD^transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpN^transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp@^transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpB^transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp@^transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpB^transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ь
_input_shapesК
З:€€€€€€€€€Љi:€€€€€€€€€::::::::::::::::::::::::2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2r
7token_and_position_embedding/embedding/embedding_lookup7token_and_position_embedding/embedding/embedding_lookup2v
9token_and_position_embedding/embedding_1/embedding_lookup9token_and_position_embedding/embedding_1/embedding_lookup2И
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpBtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp2Р
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpFtransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp2И
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpBtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp2Р
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpFtransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp2†
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpNtransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp2і
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2Ж
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOpAtransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp2Ъ
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpKtransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2К
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOpCtransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp2Ю
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpMtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2К
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOpCtransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp2Ю
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpMtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2В
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp2Ж
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpAtransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp2В
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp2Ж
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpAtransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:R N
(
_output_shapes
:€€€€€€€€€Љi
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
Ф
Ё
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
identityИҐStatefulPartitionedCallЙ
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
:€€€€€€€€€*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__wrapped_model_882872
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ь
_input_shapesК
З:€€€€€€€€€Љi:€€€€€€€€€::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€Љi
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_2
ј5
к
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
identityИҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐ4token_and_position_embedding/StatefulPartitionedCallҐ+transformer_block_1/StatefulPartitionedCallэ
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs"token_and_position_embedding_89275"token_and_position_embedding_89277*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Љi *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_8851726
4token_and_position_embedding/StatefulPartitionedCallђ
!average_pooling1d/PartitionedCallPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_882962#
!average_pooling1d/PartitionedCallЙ
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0transformer_block_1_89281transformer_block_1_89283transformer_block_1_89285transformer_block_1_89287transformer_block_1_89289transformer_block_1_89291transformer_block_1_89293transformer_block_1_89295transformer_block_1_89297transformer_block_1_89299transformer_block_1_89301transformer_block_1_89303transformer_block_1_89305transformer_block_1_89307transformer_block_1_89309transformer_block_1_89311*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_888092-
+transformer_block_1/StatefulPartitionedCallі
(global_average_pooling1d/PartitionedCallPartitionedCall4transformer_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_889232*
(global_average_pooling1d/PartitionedCallХ
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_889372
concatenate/PartitionedCall≠
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_89316dense_4_89318*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_889572!
dense_4/StatefulPartitionedCallы
dropout_4/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_889902
dropout_4/PartitionedCallЂ
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_5_89322dense_5_89324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_890142!
dense_5/StatefulPartitionedCallы
dropout_5/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_890472
dropout_5/PartitionedCallЂ
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_6_89328dense_6_89330*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_890702!
dense_6/StatefulPartitionedCall«
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ь
_input_shapesК
З:€€€€€€€€€Љi:€€€€€€€€€::::::::::::::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Љi
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ґ
я
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
identityИҐStatefulPartitionedCall©
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
:€€€€€€€€€*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_893342
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ь
_input_shapesК
З:€€€€€€€€€Љi:€€€€€€€€€::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€Љi
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_2
Р	
џ
B__inference_dense_6_layer_call_and_return_conditional_losses_89070

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
°
b
)__inference_dropout_4_layer_call_fn_90405

inputs
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_889852
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
М
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_88985

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeј
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seed*2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
•Ѓ
Я
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
identityИҐdense_4/BiasAdd/ReadVariableOpҐdense_4/MatMul/ReadVariableOpҐdense_5/BiasAdd/ReadVariableOpҐdense_5/MatMul/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐ7token_and_position_embedding/embedding/embedding_lookupҐ9token_and_position_embedding/embedding_1/embedding_lookupҐBtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpҐFtransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpҐBtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpҐFtransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpҐNtransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpҐXtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpҐAtransformer_block_1/multi_head_attention_1/key/add/ReadVariableOpҐKtransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpҐCtransformer_block_1/multi_head_attention_1/query/add/ReadVariableOpҐMtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpҐCtransformer_block_1/multi_head_attention_1/value/add/ReadVariableOpҐMtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpҐ?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpҐAtransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpҐ?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpҐAtransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpА
"token_and_position_embedding/ShapeShapeinputs_0*
T0*
_output_shapes
:2$
"token_and_position_embedding/ShapeЈ
0token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€22
0token_and_position_embedding/strided_slice/stack≤
2token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 24
2token_and_position_embedding/strided_slice/stack_1≤
2token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2token_and_position_embedding/strided_slice/stack_2Р
*token_and_position_embedding/strided_sliceStridedSlice+token_and_position_embedding/Shape:output:09token_and_position_embedding/strided_slice/stack:output:0;token_and_position_embedding/strided_slice/stack_1:output:0;token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*token_and_position_embedding/strided_sliceЦ
(token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2*
(token_and_position_embedding/range/startЦ
(token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2*
(token_and_position_embedding/range/deltaС
"token_and_position_embedding/rangeRange1token_and_position_embedding/range/start:output:03token_and_position_embedding/strided_slice:output:01token_and_position_embedding/range/delta:output:0*#
_output_shapes
:€€€€€€€€€2$
"token_and_position_embedding/rangeЊ
9token_and_position_embedding/embedding_1/embedding_lookupResourceGather?token_and_position_embedding_embedding_1_embedding_lookup_89459+token_and_position_embedding/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*R
_classH
FDloc:@token_and_position_embedding/embedding_1/embedding_lookup/89459*'
_output_shapes
:€€€€€€€€€ *
dtype02;
9token_and_position_embedding/embedding_1/embedding_lookupМ
Btoken_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityBtoken_and_position_embedding/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*R
_classH
FDloc:@token_and_position_embedding/embedding_1/embedding_lookup/89459*'
_output_shapes
:€€€€€€€€€ 2D
Btoken_and_position_embedding/embedding_1/embedding_lookup/IdentityЧ
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityKtoken_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2F
Dtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1Ѓ
+token_and_position_embedding/embedding/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€Љi2-
+token_and_position_embedding/embedding/Castњ
7token_and_position_embedding/embedding/embedding_lookupResourceGather=token_and_position_embedding_embedding_embedding_lookup_89465/token_and_position_embedding/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*P
_classF
DBloc:@token_and_position_embedding/embedding/embedding_lookup/89465*,
_output_shapes
:€€€€€€€€€Љi *
dtype029
7token_and_position_embedding/embedding/embedding_lookupЙ
@token_and_position_embedding/embedding/embedding_lookup/IdentityIdentity@token_and_position_embedding/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*P
_classF
DBloc:@token_and_position_embedding/embedding/embedding_lookup/89465*,
_output_shapes
:€€€€€€€€€Љi 2B
@token_and_position_embedding/embedding/embedding_lookup/IdentityЦ
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityItoken_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€Љi 2D
Btoken_and_position_embedding/embedding/embedding_lookup/Identity_1†
 token_and_position_embedding/addAddV2Ktoken_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Mtoken_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Љi 2"
 token_and_position_embedding/addЖ
 average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 average_pooling1d/ExpandDims/dim÷
average_pooling1d/ExpandDims
ExpandDims$token_and_position_embedding/add:z:0)average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Љi 2
average_pooling1d/ExpandDimsа
average_pooling1d/AvgPoolAvgPool%average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:€€€€€€€€€- *
ksize	
ђ*
paddingVALID*
strides	
ђ2
average_pooling1d/AvgPool≤
average_pooling1d/SqueezeSqueeze"average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€- *
squeeze_dims
2
average_pooling1d/Squeezeє
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_1_multi_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpе
>transformer_block_1/multi_head_attention_1/query/einsum/EinsumEinsum"average_pooling1d/Squeeze:output:0Utransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2@
>transformer_block_1/multi_head_attention_1/query/einsum/EinsumЧ
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOpReadVariableOpLtransformer_block_1_multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp≈
4transformer_block_1/multi_head_attention_1/query/addAddV2Gtransformer_block_1/multi_head_attention_1/query/einsum/Einsum:output:0Ktransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 26
4transformer_block_1/multi_head_attention_1/query/add≥
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_1_multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpя
<transformer_block_1/multi_head_attention_1/key/einsum/EinsumEinsum"average_pooling1d/Squeeze:output:0Stransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2>
<transformer_block_1/multi_head_attention_1/key/einsum/EinsumС
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOpReadVariableOpJtransformer_block_1_multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOpљ
2transformer_block_1/multi_head_attention_1/key/addAddV2Etransformer_block_1/multi_head_attention_1/key/einsum/Einsum:output:0Itransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 24
2transformer_block_1/multi_head_attention_1/key/addє
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_1_multi_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpе
>transformer_block_1/multi_head_attention_1/value/einsum/EinsumEinsum"average_pooling1d/Squeeze:output:0Utransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2@
>transformer_block_1/multi_head_attention_1/value/einsum/EinsumЧ
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOpReadVariableOpLtransformer_block_1_multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp≈
4transformer_block_1/multi_head_attention_1/value/addAddV2Gtransformer_block_1/multi_head_attention_1/value/einsum/Einsum:output:0Ktransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 26
4transformer_block_1/multi_head_attention_1/value/add©
0transformer_block_1/multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>22
0transformer_block_1/multi_head_attention_1/Mul/yЦ
.transformer_block_1/multi_head_attention_1/MulMul8transformer_block_1/multi_head_attention_1/query/add:z:09transformer_block_1/multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€- 20
.transformer_block_1/multi_head_attention_1/Mulћ
8transformer_block_1/multi_head_attention_1/einsum/EinsumEinsum6transformer_block_1/multi_head_attention_1/key/add:z:02transformer_block_1/multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€--*
equationaecd,abcd->acbe2:
8transformer_block_1/multi_head_attention_1/einsum/EinsumА
:transformer_block_1/multi_head_attention_1/softmax/SoftmaxSoftmaxAtransformer_block_1/multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€--2<
:transformer_block_1/multi_head_attention_1/softmax/Softmax…
@transformer_block_1/multi_head_attention_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2B
@transformer_block_1/multi_head_attention_1/dropout/dropout/Const“
>transformer_block_1/multi_head_attention_1/dropout/dropout/MulMulDtransformer_block_1/multi_head_attention_1/softmax/Softmax:softmax:0Itransformer_block_1/multi_head_attention_1/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€--2@
>transformer_block_1/multi_head_attention_1/dropout/dropout/Mulш
@transformer_block_1/multi_head_attention_1/dropout/dropout/ShapeShapeDtransformer_block_1/multi_head_attention_1/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2B
@transformer_block_1/multi_head_attention_1/dropout/dropout/Shapeб
Wtransformer_block_1/multi_head_attention_1/dropout/dropout/random_uniform/RandomUniformRandomUniformItransformer_block_1/multi_head_attention_1/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€--*
dtype0*

seed*2Y
Wtransformer_block_1/multi_head_attention_1/dropout/dropout/random_uniform/RandomUniformџ
Itransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Itransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqual/yТ
Gtransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqualGreaterEqual`transformer_block_1/multi_head_attention_1/dropout/dropout/random_uniform/RandomUniform:output:0Rtransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€--2I
Gtransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqual†
?transformer_block_1/multi_head_attention_1/dropout/dropout/CastCastKtransformer_block_1/multi_head_attention_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€--2A
?transformer_block_1/multi_head_attention_1/dropout/dropout/Castќ
@transformer_block_1/multi_head_attention_1/dropout/dropout/Mul_1MulBtransformer_block_1/multi_head_attention_1/dropout/dropout/Mul:z:0Ctransformer_block_1/multi_head_attention_1/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€--2B
@transformer_block_1/multi_head_attention_1/dropout/dropout/Mul_1д
:transformer_block_1/multi_head_attention_1/einsum_1/EinsumEinsumDtransformer_block_1/multi_head_attention_1/dropout/dropout/Mul_1:z:08transformer_block_1/multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationacbe,aecd->abcd2<
:transformer_block_1/multi_head_attention_1/einsum_1/EinsumЏ
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_1_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp£
Itransformer_block_1/multi_head_attention_1/attention_output/einsum/EinsumEinsumCtransformer_block_1/multi_head_attention_1/einsum_1/Einsum:output:0`transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€- *
equationabcd,cde->abe2K
Itransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsumі
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_1_multi_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpн
?transformer_block_1/multi_head_attention_1/attention_output/addAddV2Rtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum:output:0Vtransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2A
?transformer_block_1/multi_head_attention_1/attention_output/addЯ
+transformer_block_1/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2-
+transformer_block_1/dropout_2/dropout/ConstО
)transformer_block_1/dropout_2/dropout/MulMulCtransformer_block_1/multi_head_attention_1/attention_output/add:z:04transformer_block_1/dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2+
)transformer_block_1/dropout_2/dropout/MulЌ
+transformer_block_1/dropout_2/dropout/ShapeShapeCtransformer_block_1/multi_head_attention_1/attention_output/add:z:0*
T0*
_output_shapes
:2-
+transformer_block_1/dropout_2/dropout/ShapeЂ
Btransformer_block_1/dropout_2/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_1/dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€- *
dtype0*

seed**
seed22D
Btransformer_block_1/dropout_2/dropout/random_uniform/RandomUniform±
4transformer_block_1/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=26
4transformer_block_1/dropout_2/dropout/GreaterEqual/yЇ
2transformer_block_1/dropout_2/dropout/GreaterEqualGreaterEqualKtransformer_block_1/dropout_2/dropout/random_uniform/RandomUniform:output:0=transformer_block_1/dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 24
2transformer_block_1/dropout_2/dropout/GreaterEqualЁ
*transformer_block_1/dropout_2/dropout/CastCast6transformer_block_1/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€- 2,
*transformer_block_1/dropout_2/dropout/Castц
+transformer_block_1/dropout_2/dropout/Mul_1Mul-transformer_block_1/dropout_2/dropout/Mul:z:0.transformer_block_1/dropout_2/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€- 2-
+transformer_block_1/dropout_2/dropout/Mul_1∆
transformer_block_1/addAddV2"average_pooling1d/Squeeze:output:0/transformer_block_1/dropout_2/dropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
transformer_block_1/addё
Htransformer_block_1/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_1/layer_normalization_2/moments/mean/reduction_indicesѓ
6transformer_block_1/layer_normalization_2/moments/meanMeantransformer_block_1/add:z:0Qtransformer_block_1/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(28
6transformer_block_1/layer_normalization_2/moments/meanЗ
>transformer_block_1/layer_normalization_2/moments/StopGradientStopGradient?transformer_block_1/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2@
>transformer_block_1/layer_normalization_2/moments/StopGradientї
Ctransformer_block_1/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencetransformer_block_1/add:z:0Gtransformer_block_1/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2E
Ctransformer_block_1/layer_normalization_2/moments/SquaredDifferenceж
Ltransformer_block_1/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_1/layer_normalization_2/moments/variance/reduction_indicesз
:transformer_block_1/layer_normalization_2/moments/varianceMeanGtransformer_block_1/layer_normalization_2/moments/SquaredDifference:z:0Utransformer_block_1/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2<
:transformer_block_1/layer_normalization_2/moments/varianceї
9transformer_block_1/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52;
9transformer_block_1/layer_normalization_2/batchnorm/add/yЇ
7transformer_block_1/layer_normalization_2/batchnorm/addAddV2Ctransformer_block_1/layer_normalization_2/moments/variance:output:0Btransformer_block_1/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€-29
7transformer_block_1/layer_normalization_2/batchnorm/addт
9transformer_block_1/layer_normalization_2/batchnorm/RsqrtRsqrt;transformer_block_1/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€-2;
9transformer_block_1/layer_normalization_2/batchnorm/RsqrtЬ
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpЊ
7transformer_block_1/layer_normalization_2/batchnorm/mulMul=transformer_block_1/layer_normalization_2/batchnorm/Rsqrt:y:0Ntransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 29
7transformer_block_1/layer_normalization_2/batchnorm/mulН
9transformer_block_1/layer_normalization_2/batchnorm/mul_1Multransformer_block_1/add:z:0;transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2;
9transformer_block_1/layer_normalization_2/batchnorm/mul_1±
9transformer_block_1/layer_normalization_2/batchnorm/mul_2Mul?transformer_block_1/layer_normalization_2/moments/mean:output:0;transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2;
9transformer_block_1/layer_normalization_2/batchnorm/mul_2Р
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_1_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpЇ
7transformer_block_1/layer_normalization_2/batchnorm/subSubJtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp:value:0=transformer_block_1/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 29
7transformer_block_1/layer_normalization_2/batchnorm/sub±
9transformer_block_1/layer_normalization_2/batchnorm/add_1AddV2=transformer_block_1/layer_normalization_2/batchnorm/mul_1:z:0;transformer_block_1/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2;
9transformer_block_1/layer_normalization_2/batchnorm/add_1С
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_1_sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02C
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpЉ
7transformer_block_1/sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_1/sequential_1/dense_2/Tensordot/axes√
7transformer_block_1/sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_1/sequential_1/dense_2/Tensordot/freeб
8transformer_block_1/sequential_1/dense_2/Tensordot/ShapeShape=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_2/Tensordot/Shape∆
@transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axisЮ
;transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2GatherV2Atransformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2 
Btransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axis§
=transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1GatherV2Atransformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Ktransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1Њ
8transformer_block_1/sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_1/sequential_1/dense_2/Tensordot/Const§
7transformer_block_1/sequential_1/dense_2/Tensordot/ProdProdDtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Atransformer_block_1/sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_1/sequential_1/dense_2/Tensordot/Prod¬
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_1ђ
9transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1ProdFtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1:output:0Ctransformer_block_1/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1¬
>transformer_block_1/sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_1/sequential_1/dense_2/Tensordot/concat/axisэ
9transformer_block_1/sequential_1/dense_2/Tensordot/concatConcatV2@transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0@transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Gtransformer_block_1/sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_1/sequential_1/dense_2/Tensordot/concat∞
8transformer_block_1/sequential_1/dense_2/Tensordot/stackPack@transformer_block_1/sequential_1/dense_2/Tensordot/Prod:output:0Btransformer_block_1/sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_2/Tensordot/stack¬
<transformer_block_1/sequential_1/dense_2/Tensordot/transpose	Transpose=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0Btransformer_block_1/sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2>
<transformer_block_1/sequential_1/dense_2/Tensordot/transpose√
:transformer_block_1/sequential_1/dense_2/Tensordot/ReshapeReshape@transformer_block_1/sequential_1/dense_2/Tensordot/transpose:y:0Atransformer_block_1/sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2<
:transformer_block_1/sequential_1/dense_2/Tensordot/Reshape¬
9transformer_block_1/sequential_1/dense_2/Tensordot/MatMulMatMulCtransformer_block_1/sequential_1/dense_2/Tensordot/Reshape:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2;
9transformer_block_1/sequential_1/dense_2/Tensordot/MatMul¬
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_2/Tensordot/Const_2∆
@transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axisК
;transformer_block_1/sequential_1/dense_2/Tensordot/concat_1ConcatV2Dtransformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Ctransformer_block_1/sequential_1/dense_2/Tensordot/Const_2:output:0Itransformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_2/Tensordot/concat_1і
2transformer_block_1/sequential_1/dense_2/TensordotReshapeCtransformer_block_1/sequential_1/dense_2/Tensordot/MatMul:product:0Dtransformer_block_1/sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 24
2transformer_block_1/sequential_1/dense_2/TensordotЗ
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_1_sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpЂ
0transformer_block_1/sequential_1/dense_2/BiasAddBiasAdd;transformer_block_1/sequential_1/dense_2/Tensordot:output:0Gtransformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 22
0transformer_block_1/sequential_1/dense_2/BiasAdd„
-transformer_block_1/sequential_1/dense_2/ReluRelu9transformer_block_1/sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2/
-transformer_block_1/sequential_1/dense_2/ReluС
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOpJtransformer_block_1_sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02C
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpЉ
7transformer_block_1/sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:29
7transformer_block_1/sequential_1/dense_3/Tensordot/axes√
7transformer_block_1/sequential_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       29
7transformer_block_1/sequential_1/dense_3/Tensordot/freeя
8transformer_block_1/sequential_1/dense_3/Tensordot/ShapeShape;transformer_block_1/sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_3/Tensordot/Shape∆
@transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axisЮ
;transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2GatherV2Atransformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2 
Btransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axis§
=transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1GatherV2Atransformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Ktransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1Њ
8transformer_block_1/sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8transformer_block_1/sequential_1/dense_3/Tensordot/Const§
7transformer_block_1/sequential_1/dense_3/Tensordot/ProdProdDtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Atransformer_block_1/sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 29
7transformer_block_1/sequential_1/dense_3/Tensordot/Prod¬
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_1ђ
9transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1ProdFtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1:output:0Ctransformer_block_1/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2;
9transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1¬
>transformer_block_1/sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block_1/sequential_1/dense_3/Tensordot/concat/axisэ
9transformer_block_1/sequential_1/dense_3/Tensordot/concatConcatV2@transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0@transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Gtransformer_block_1/sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_1/sequential_1/dense_3/Tensordot/concat∞
8transformer_block_1/sequential_1/dense_3/Tensordot/stackPack@transformer_block_1/sequential_1/dense_3/Tensordot/Prod:output:0Btransformer_block_1/sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_1/sequential_1/dense_3/Tensordot/stackј
<transformer_block_1/sequential_1/dense_3/Tensordot/transpose	Transpose;transformer_block_1/sequential_1/dense_2/Relu:activations:0Btransformer_block_1/sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2>
<transformer_block_1/sequential_1/dense_3/Tensordot/transpose√
:transformer_block_1/sequential_1/dense_3/Tensordot/ReshapeReshape@transformer_block_1/sequential_1/dense_3/Tensordot/transpose:y:0Atransformer_block_1/sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2<
:transformer_block_1/sequential_1/dense_3/Tensordot/Reshape¬
9transformer_block_1/sequential_1/dense_3/Tensordot/MatMulMatMulCtransformer_block_1/sequential_1/dense_3/Tensordot/Reshape:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2;
9transformer_block_1/sequential_1/dense_3/Tensordot/MatMul¬
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_1/sequential_1/dense_3/Tensordot/Const_2∆
@transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axisК
;transformer_block_1/sequential_1/dense_3/Tensordot/concat_1ConcatV2Dtransformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Ctransformer_block_1/sequential_1/dense_3/Tensordot/Const_2:output:0Itransformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_1/sequential_1/dense_3/Tensordot/concat_1і
2transformer_block_1/sequential_1/dense_3/TensordotReshapeCtransformer_block_1/sequential_1/dense_3/Tensordot/MatMul:product:0Dtransformer_block_1/sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 24
2transformer_block_1/sequential_1/dense_3/TensordotЗ
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOpHtransformer_block_1_sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpЂ
0transformer_block_1/sequential_1/dense_3/BiasAddBiasAdd;transformer_block_1/sequential_1/dense_3/Tensordot:output:0Gtransformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 22
0transformer_block_1/sequential_1/dense_3/BiasAddЯ
+transformer_block_1/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2-
+transformer_block_1/dropout_3/dropout/ConstД
)transformer_block_1/dropout_3/dropout/MulMul9transformer_block_1/sequential_1/dense_3/BiasAdd:output:04transformer_block_1/dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2+
)transformer_block_1/dropout_3/dropout/Mul√
+transformer_block_1/dropout_3/dropout/ShapeShape9transformer_block_1/sequential_1/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2-
+transformer_block_1/dropout_3/dropout/ShapeЂ
Btransformer_block_1/dropout_3/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_1/dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€- *
dtype0*

seed**
seed22D
Btransformer_block_1/dropout_3/dropout/random_uniform/RandomUniform±
4transformer_block_1/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=26
4transformer_block_1/dropout_3/dropout/GreaterEqual/yЇ
2transformer_block_1/dropout_3/dropout/GreaterEqualGreaterEqualKtransformer_block_1/dropout_3/dropout/random_uniform/RandomUniform:output:0=transformer_block_1/dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 24
2transformer_block_1/dropout_3/dropout/GreaterEqualЁ
*transformer_block_1/dropout_3/dropout/CastCast6transformer_block_1/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€- 2,
*transformer_block_1/dropout_3/dropout/Castц
+transformer_block_1/dropout_3/dropout/Mul_1Mul-transformer_block_1/dropout_3/dropout/Mul:z:0.transformer_block_1/dropout_3/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€- 2-
+transformer_block_1/dropout_3/dropout/Mul_1е
transformer_block_1/add_1AddV2=transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0/transformer_block_1/dropout_3/dropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
transformer_block_1/add_1ё
Htransformer_block_1/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_1/layer_normalization_3/moments/mean/reduction_indices±
6transformer_block_1/layer_normalization_3/moments/meanMeantransformer_block_1/add_1:z:0Qtransformer_block_1/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(28
6transformer_block_1/layer_normalization_3/moments/meanЗ
>transformer_block_1/layer_normalization_3/moments/StopGradientStopGradient?transformer_block_1/layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2@
>transformer_block_1/layer_normalization_3/moments/StopGradientљ
Ctransformer_block_1/layer_normalization_3/moments/SquaredDifferenceSquaredDifferencetransformer_block_1/add_1:z:0Gtransformer_block_1/layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2E
Ctransformer_block_1/layer_normalization_3/moments/SquaredDifferenceж
Ltransformer_block_1/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_1/layer_normalization_3/moments/variance/reduction_indicesз
:transformer_block_1/layer_normalization_3/moments/varianceMeanGtransformer_block_1/layer_normalization_3/moments/SquaredDifference:z:0Utransformer_block_1/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2<
:transformer_block_1/layer_normalization_3/moments/varianceї
9transformer_block_1/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52;
9transformer_block_1/layer_normalization_3/batchnorm/add/yЇ
7transformer_block_1/layer_normalization_3/batchnorm/addAddV2Ctransformer_block_1/layer_normalization_3/moments/variance:output:0Btransformer_block_1/layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€-29
7transformer_block_1/layer_normalization_3/batchnorm/addт
9transformer_block_1/layer_normalization_3/batchnorm/RsqrtRsqrt;transformer_block_1/layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€-2;
9transformer_block_1/layer_normalization_3/batchnorm/RsqrtЬ
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpЊ
7transformer_block_1/layer_normalization_3/batchnorm/mulMul=transformer_block_1/layer_normalization_3/batchnorm/Rsqrt:y:0Ntransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 29
7transformer_block_1/layer_normalization_3/batchnorm/mulП
9transformer_block_1/layer_normalization_3/batchnorm/mul_1Multransformer_block_1/add_1:z:0;transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2;
9transformer_block_1/layer_normalization_3/batchnorm/mul_1±
9transformer_block_1/layer_normalization_3/batchnorm/mul_2Mul?transformer_block_1/layer_normalization_3/moments/mean:output:0;transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2;
9transformer_block_1/layer_normalization_3/batchnorm/mul_2Р
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpЇ
7transformer_block_1/layer_normalization_3/batchnorm/subSubJtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp:value:0=transformer_block_1/layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 29
7transformer_block_1/layer_normalization_3/batchnorm/sub±
9transformer_block_1/layer_normalization_3/batchnorm/add_1AddV2=transformer_block_1/layer_normalization_3/batchnorm/mul_1:z:0;transformer_block_1/layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2;
9transformer_block_1/layer_normalization_3/batchnorm/add_1§
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indicesс
global_average_pooling1d/MeanMean=transformer_block_1/layer_normalization_3/batchnorm/add_1:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
global_average_pooling1d/Meant
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis√
concatenate/concatConcatV2&global_average_pooling1d/Mean:output:0inputs_1 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€(2
concatenate/concat•
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:( *
dtype02
dense_4/MatMul/ReadVariableOp†
dense_4/MatMulMatMulconcatenate/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_4/MatMul§
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOp°
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_4/Reluw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_4/dropout/Const•
dropout_4/dropout/MulMuldense_4/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_4/dropout/Mul|
dropout_4/dropout/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shapeл
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seed**
seed220
.dropout_4/dropout/random_uniform/RandomUniformЙ
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_4/dropout/GreaterEqual/yж
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
dropout_4/dropout/GreaterEqualЭ
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout_4/dropout/CastҐ
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_4/dropout/Mul_1•
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_5/MatMul/ReadVariableOp†
dense_5/MatMulMatMuldropout_4/dropout/Mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_5/MatMul§
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_5/BiasAdd/ReadVariableOp°
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_5/Reluw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_5/dropout/Const•
dropout_5/dropout/MulMuldense_5/Relu:activations:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_5/dropout/Mul|
dropout_5/dropout/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shapeл
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seed**
seed220
.dropout_5/dropout/random_uniform/RandomUniformЙ
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_5/dropout/GreaterEqual/yж
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
dropout_5/dropout/GreaterEqualЭ
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout_5/dropout/CastҐ
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_5/dropout/Mul_1•
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_6/MatMul/ReadVariableOp†
dense_6/MatMulMatMuldropout_5/dropout/Mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_6/MatMul§
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp°
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_6/BiasAddЈ
IdentityIdentitydense_6/BiasAdd:output:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp8^token_and_position_embedding/embedding/embedding_lookup:^token_and_position_embedding/embedding_1/embedding_lookupC^transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpG^transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpC^transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpG^transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpO^transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpY^transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_1/multi_head_attention_1/key/add/ReadVariableOpL^transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpD^transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpN^transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpD^transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpN^transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp@^transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpB^transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp@^transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpB^transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ь
_input_shapesК
З:€€€€€€€€€Љi:€€€€€€€€€::::::::::::::::::::::::2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2r
7token_and_position_embedding/embedding/embedding_lookup7token_and_position_embedding/embedding/embedding_lookup2v
9token_and_position_embedding/embedding_1/embedding_lookup9token_and_position_embedding/embedding_1/embedding_lookup2И
Btransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpBtransformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp2Р
Ftransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpFtransformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp2И
Btransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpBtransformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp2Р
Ftransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpFtransformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp2†
Ntransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpNtransformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp2і
Xtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2Ж
Atransformer_block_1/multi_head_attention_1/key/add/ReadVariableOpAtransformer_block_1/multi_head_attention_1/key/add/ReadVariableOp2Ъ
Ktransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpKtransformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2К
Ctransformer_block_1/multi_head_attention_1/query/add/ReadVariableOpCtransformer_block_1/multi_head_attention_1/query/add/ReadVariableOp2Ю
Mtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpMtransformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2К
Ctransformer_block_1/multi_head_attention_1/value/add/ReadVariableOpCtransformer_block_1/multi_head_attention_1/value/add/ReadVariableOp2Ю
Mtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpMtransformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2В
?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp?transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp2Ж
Atransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpAtransformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp2В
?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp?transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp2Ж
Atransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpAtransformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:R N
(
_output_shapes
:€€€€€€€€€Љi
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
ЎH
¶
G__inference_sequential_1_layer_call_and_return_conditional_losses_90533

inputs-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource-
)dense_3_tensordot_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityИҐdense_2/BiasAdd/ReadVariableOpҐ dense_2/Tensordot/ReadVariableOpҐdense_3/BiasAdd/ReadVariableOpҐ dense_3/Tensordot/ReadVariableOpЃ
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
dense_2/Tensordot/axesБ
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
dense_2/Tensordot/ShapeД
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axisщ
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2И
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis€
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
dense_2/Tensordot/Const†
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/ProdА
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1®
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1А
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axisЎ
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concatђ
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack®
dense_2/Tensordot/transpose	Transposeinputs!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dense_2/Tensordot/transposeњ
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_2/Tensordot/ReshapeЊ
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_2/Tensordot/MatMulА
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_2Д
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axisе
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1∞
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dense_2/Tensordot§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_2/BiasAdd/ReadVariableOpІ
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dense_2/BiasAddt
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dense_2/ReluЃ
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
dense_3/Tensordot/axesБ
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
dense_3/Tensordot/ShapeД
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/GatherV2/axisщ
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2И
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axis€
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
dense_3/Tensordot/Const†
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/ProdА
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1®
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1А
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axisЎ
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concatђ
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stackЉ
dense_3/Tensordot/transpose	Transposedense_2/Relu:activations:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dense_3/Tensordot/transposeњ
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_3/Tensordot/ReshapeЊ
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_3/Tensordot/MatMulА
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_2Д
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axisе
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1∞
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dense_3/Tensordot§
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOpІ
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dense_3/BiasAddш
IdentityIdentitydense_3/BiasAdd:output:0^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€- ::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€- 
 
_user_specified_nameinputs
≤
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
:€€€€€€€€€(2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€ :€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ѕ
ы
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_89970
x&
"embedding_1_embedding_lookup_89957$
 embedding_embedding_lookup_89963
identityИҐembedding/embedding_lookupҐembedding_1/embedding_lookup?
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
€€€€€€€€€2
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
strided_slice/stack_2в
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
range/deltaА
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:€€€€€€€€€2
range≠
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_89957range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/89957*'
_output_shapes
:€€€€€€€€€ *
dtype02
embedding_1/embedding_lookupШ
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/89957*'
_output_shapes
:€€€€€€€€€ 2'
%embedding_1/embedding_lookup/Identityј
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'embedding_1/embedding_lookup/Identity_1m
embedding/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€Љi2
embedding/CastЃ
embedding/embedding_lookupResourceGather embedding_embedding_lookup_89963embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/89963*,
_output_shapes
:€€€€€€€€€Љi *
dtype02
embedding/embedding_lookupХ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/89963*,
_output_shapes
:€€€€€€€€€Љi 2%
#embedding/embedding_lookup/Identityњ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€Љi 2'
%embedding/embedding_lookup/Identity_1ђ
addAddV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Љi 2
addЬ
IdentityIdentityadd:z:0^embedding/embedding_lookup^embedding_1/embedding_lookup*
T0*,
_output_shapes
:€€€€€€€€€Љi 2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Љi::28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:K G
(
_output_shapes
:€€€€€€€€€Љi

_user_specified_namex
А№
—
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
identityИҐ.layer_normalization_2/batchnorm/ReadVariableOpҐ2layer_normalization_2/batchnorm/mul/ReadVariableOpҐ.layer_normalization_3/batchnorm/ReadVariableOpҐ2layer_normalization_3/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_1/attention_output/add/ReadVariableOpҐDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_1/key/add/ReadVariableOpҐ7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_1/query/add/ReadVariableOpҐ9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_1/value/add/ReadVariableOpҐ9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpҐ+sequential_1/dense_2/BiasAdd/ReadVariableOpҐ-sequential_1/dense_2/Tensordot/ReadVariableOpҐ+sequential_1/dense_3/BiasAdd/ReadVariableOpҐ-sequential_1/dense_3/Tensordot/ReadVariableOpэ
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpН
*multi_head_attention_1/query/einsum/EinsumEinsuminputsAmulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/Einsumџ
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/query/add/ReadVariableOpх
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 2"
 multi_head_attention_1/query/addч
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpЗ
(multi_head_attention_1/key/einsum/EinsumEinsuminputs?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/Einsum’
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_1/key/add/ReadVariableOpн
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 2 
multi_head_attention_1/key/addэ
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpН
*multi_head_attention_1/value/einsum/EinsumEinsuminputsAmulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/Einsumџ
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/value/add/ReadVariableOpх
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 2"
 multi_head_attention_1/value/addБ
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2
multi_head_attention_1/Mul/y∆
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€- 2
multi_head_attention_1/Mulь
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€--*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/Einsumƒ
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€--2(
&multi_head_attention_1/softmax/Softmax 
'multi_head_attention_1/dropout/IdentityIdentity0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€--2)
'multi_head_attention_1/dropout/IdentityФ
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/Identity:output:0$multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/EinsumЮ
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp”
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€- *
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/Einsumш
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOpЭ
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2-
+multi_head_attention_1/attention_output/addЫ
dropout_2/IdentityIdentity/multi_head_attention_1/attention_output/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dropout_2/Identityn
addAddV2inputsdropout_2/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
addґ
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indicesя
"layer_normalization_2/moments/meanMeanadd:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2$
"layer_normalization_2/moments/meanЋ
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2,
*layer_normalization_2/moments/StopGradientл
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 21
/layer_normalization_2/moments/SquaredDifferenceЊ
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indicesЧ
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2(
&layer_normalization_2/moments/varianceУ
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_2/batchnorm/add/yк
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2%
#layer_normalization_2/batchnorm/addґ
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€-2'
%layer_normalization_2/batchnorm/Rsqrtа
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOpо
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2%
#layer_normalization_2/batchnorm/mulљ
%layer_normalization_2/batchnorm/mul_1Muladd:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_2/batchnorm/mul_1б
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_2/batchnorm/mul_2‘
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_2/batchnorm/ReadVariableOpк
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2%
#layer_normalization_2/batchnorm/subб
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_2/batchnorm/add_1’
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_1/dense_2/Tensordot/ReadVariableOpФ
#sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_2/Tensordot/axesЫ
#sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_2/Tensordot/free•
$sequential_1/dense_2/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/ShapeЮ
,sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/GatherV2/axisЇ
'sequential_1/dense_2/Tensordot/GatherV2GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/free:output:05sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/GatherV2Ґ
.sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_2/Tensordot/GatherV2_1/axisј
)sequential_1/dense_2/Tensordot/GatherV2_1GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/axes:output:07sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_2/Tensordot/GatherV2_1Ц
$sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_2/Tensordot/Const‘
#sequential_1/dense_2/Tensordot/ProdProd0sequential_1/dense_2/Tensordot/GatherV2:output:0-sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_2/Tensordot/ProdЪ
&sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_1№
%sequential_1/dense_2/Tensordot/Prod_1Prod2sequential_1/dense_2/Tensordot/GatherV2_1:output:0/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_2/Tensordot/Prod_1Ъ
*sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_2/Tensordot/concat/axisЩ
%sequential_1/dense_2/Tensordot/concatConcatV2,sequential_1/dense_2/Tensordot/free:output:0,sequential_1/dense_2/Tensordot/axes:output:03sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_2/Tensordot/concatа
$sequential_1/dense_2/Tensordot/stackPack,sequential_1/dense_2/Tensordot/Prod:output:0.sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/stackт
(sequential_1/dense_2/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0.sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2*
(sequential_1/dense_2/Tensordot/transposeу
&sequential_1/dense_2/Tensordot/ReshapeReshape,sequential_1/dense_2/Tensordot/transpose:y:0-sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_1/dense_2/Tensordot/Reshapeт
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_1/dense_2/Tensordot/MatMulЪ
&sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_2Ю
,sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/concat_1/axis¶
'sequential_1/dense_2/Tensordot/concat_1ConcatV20sequential_1/dense_2/Tensordot/GatherV2:output:0/sequential_1/dense_2/Tensordot/Const_2:output:05sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/concat_1д
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:00sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2 
sequential_1/dense_2/TensordotЋ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOpџ
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
sequential_1/dense_2/BiasAddЫ
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
sequential_1/dense_2/Relu’
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_1/dense_3/Tensordot/ReadVariableOpФ
#sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_3/Tensordot/axesЫ
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
$sequential_1/dense_3/Tensordot/ShapeЮ
,sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/GatherV2/axisЇ
'sequential_1/dense_3/Tensordot/GatherV2GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/free:output:05sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/GatherV2Ґ
.sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_3/Tensordot/GatherV2_1/axisј
)sequential_1/dense_3/Tensordot/GatherV2_1GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/axes:output:07sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_3/Tensordot/GatherV2_1Ц
$sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_3/Tensordot/Const‘
#sequential_1/dense_3/Tensordot/ProdProd0sequential_1/dense_3/Tensordot/GatherV2:output:0-sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_3/Tensordot/ProdЪ
&sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_1№
%sequential_1/dense_3/Tensordot/Prod_1Prod2sequential_1/dense_3/Tensordot/GatherV2_1:output:0/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_3/Tensordot/Prod_1Ъ
*sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_3/Tensordot/concat/axisЩ
%sequential_1/dense_3/Tensordot/concatConcatV2,sequential_1/dense_3/Tensordot/free:output:0,sequential_1/dense_3/Tensordot/axes:output:03sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_3/Tensordot/concatа
$sequential_1/dense_3/Tensordot/stackPack,sequential_1/dense_3/Tensordot/Prod:output:0.sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/stackр
(sequential_1/dense_3/Tensordot/transpose	Transpose'sequential_1/dense_2/Relu:activations:0.sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2*
(sequential_1/dense_3/Tensordot/transposeу
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_3/Tensordot/transpose:y:0-sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_1/dense_3/Tensordot/Reshapeт
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_1/dense_3/Tensordot/MatMulЪ
&sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_2Ю
,sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/concat_1/axis¶
'sequential_1/dense_3/Tensordot/concat_1ConcatV20sequential_1/dense_3/Tensordot/GatherV2:output:0/sequential_1/dense_3/Tensordot/Const_2:output:05sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/concat_1д
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:00sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2 
sequential_1/dense_3/TensordotЋ
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOpџ
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
sequential_1/dense_3/BiasAddС
dropout_3/IdentityIdentity%sequential_1/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dropout_3/IdentityХ
add_1AddV2)layer_normalization_2/batchnorm/add_1:z:0dropout_3/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
add_1ґ
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indicesб
"layer_normalization_3/moments/meanMean	add_1:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2$
"layer_normalization_3/moments/meanЋ
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2,
*layer_normalization_3/moments/StopGradientн
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 21
/layer_normalization_3/moments/SquaredDifferenceЊ
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indicesЧ
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2(
&layer_normalization_3/moments/varianceУ
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_3/batchnorm/add/yк
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2%
#layer_normalization_3/batchnorm/addґ
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€-2'
%layer_normalization_3/batchnorm/Rsqrtа
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOpо
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2%
#layer_normalization_3/batchnorm/mulњ
%layer_normalization_3/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_3/batchnorm/mul_1б
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_3/batchnorm/mul_2‘
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_3/batchnorm/ReadVariableOpк
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2%
#layer_normalization_3/batchnorm/subб
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_3/batchnorm/add_1”
IdentityIdentity)layer_normalization_3/batchnorm/add_1:z:0/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€- ::::::::::::::::2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2М
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
:€€€€€€€€€- 
 
_user_specified_nameinputs
ƒ8
≤
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
identityИҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐ!dropout_4/StatefulPartitionedCallҐ!dropout_5/StatefulPartitionedCallҐ4token_and_position_embedding/StatefulPartitionedCallҐ+transformer_block_1/StatefulPartitionedCallэ
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs"token_and_position_embedding_89158"token_and_position_embedding_89160*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Љi *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_8851726
4token_and_position_embedding/StatefulPartitionedCallђ
!average_pooling1d/PartitionedCallPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_882962#
!average_pooling1d/PartitionedCallЙ
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0transformer_block_1_89164transformer_block_1_89166transformer_block_1_89168transformer_block_1_89170transformer_block_1_89172transformer_block_1_89174transformer_block_1_89176transformer_block_1_89178transformer_block_1_89180transformer_block_1_89182transformer_block_1_89184transformer_block_1_89186transformer_block_1_89188transformer_block_1_89190transformer_block_1_89192transformer_block_1_89194*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_886822-
+transformer_block_1/StatefulPartitionedCallі
(global_average_pooling1d/PartitionedCallPartitionedCall4transformer_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_889232*
(global_average_pooling1d/PartitionedCallХ
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_889372
concatenate/PartitionedCall≠
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_89199dense_4_89201*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_889572!
dense_4/StatefulPartitionedCallУ
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_889852#
!dropout_4/StatefulPartitionedCall≥
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_5_89205dense_5_89207*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_890142!
dense_5/StatefulPartitionedCallЈ
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_890422#
!dropout_5/StatefulPartitionedCall≥
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_6_89211dense_6_89213*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_890702!
dense_6/StatefulPartitionedCallП
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ь
_input_shapesК
З:€€€€€€€€€Љi:€€€€€€€€€::::::::::::::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Љi
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
П
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
:€€€€€€€€€€€€€€€€€€2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
џ
|
'__inference_dense_6_layer_call_fn_90476

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_890702
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Г
М
<__inference_token_and_position_embedding_layer_call_fn_89979
x
unknown
	unknown_0
identityИҐStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Љi *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_885172
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€Љi 2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Љi::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:€€€€€€€€€Љi

_user_specified_namex
мь
—
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
identityИҐ.layer_normalization_2/batchnorm/ReadVariableOpҐ2layer_normalization_2/batchnorm/mul/ReadVariableOpҐ.layer_normalization_3/batchnorm/ReadVariableOpҐ2layer_normalization_3/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_1/attention_output/add/ReadVariableOpҐDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_1/key/add/ReadVariableOpҐ7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_1/query/add/ReadVariableOpҐ9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_1/value/add/ReadVariableOpҐ9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpҐ+sequential_1/dense_2/BiasAdd/ReadVariableOpҐ-sequential_1/dense_2/Tensordot/ReadVariableOpҐ+sequential_1/dense_3/BiasAdd/ReadVariableOpҐ-sequential_1/dense_3/Tensordot/ReadVariableOpэ
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpН
*multi_head_attention_1/query/einsum/EinsumEinsuminputsAmulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/Einsumџ
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/query/add/ReadVariableOpх
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 2"
 multi_head_attention_1/query/addч
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpЗ
(multi_head_attention_1/key/einsum/EinsumEinsuminputs?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/Einsum’
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_1/key/add/ReadVariableOpн
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 2 
multi_head_attention_1/key/addэ
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpН
*multi_head_attention_1/value/einsum/EinsumEinsuminputsAmulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/Einsumџ
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/value/add/ReadVariableOpх
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 2"
 multi_head_attention_1/value/addБ
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2
multi_head_attention_1/Mul/y∆
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€- 2
multi_head_attention_1/Mulь
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€--*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/Einsumƒ
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€--2(
&multi_head_attention_1/softmax/Softmax°
,multi_head_attention_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,multi_head_attention_1/dropout/dropout/ConstВ
*multi_head_attention_1/dropout/dropout/MulMul0multi_head_attention_1/softmax/Softmax:softmax:05multi_head_attention_1/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€--2,
*multi_head_attention_1/dropout/dropout/MulЉ
,multi_head_attention_1/dropout/dropout/ShapeShape0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_1/dropout/dropout/Shape•
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_1/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€--*
dtype0*

seed*2E
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform≥
5multi_head_attention_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_1/dropout/dropout/GreaterEqual/y¬
3multi_head_attention_1/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_1/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€--25
3multi_head_attention_1/dropout/dropout/GreaterEqualд
+multi_head_attention_1/dropout/dropout/CastCast7multi_head_attention_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€--2-
+multi_head_attention_1/dropout/dropout/Castю
,multi_head_attention_1/dropout/dropout/Mul_1Mul.multi_head_attention_1/dropout/dropout/Mul:z:0/multi_head_attention_1/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€--2.
,multi_head_attention_1/dropout/dropout/Mul_1Ф
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/dropout/Mul_1:z:0$multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/EinsumЮ
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp”
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€- *
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/Einsumш
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOpЭ
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2-
+multi_head_attention_1/attention_output/addw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_2/dropout/ConstЊ
dropout_2/dropout/MulMul/multi_head_attention_1/attention_output/add:z:0 dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dropout_2/dropout/MulС
dropout_2/dropout/ShapeShape/multi_head_attention_1/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shapeп
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€- *
dtype0*

seed**
seed220
.dropout_2/dropout/random_uniform/RandomUniformЙ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_2/dropout/GreaterEqual/yк
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2 
dropout_2/dropout/GreaterEqual°
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€- 2
dropout_2/dropout/Cast¶
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dropout_2/dropout/Mul_1n
addAddV2inputsdropout_2/dropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
addґ
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indicesя
"layer_normalization_2/moments/meanMeanadd:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2$
"layer_normalization_2/moments/meanЋ
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2,
*layer_normalization_2/moments/StopGradientл
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 21
/layer_normalization_2/moments/SquaredDifferenceЊ
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indicesЧ
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2(
&layer_normalization_2/moments/varianceУ
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_2/batchnorm/add/yк
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2%
#layer_normalization_2/batchnorm/addґ
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€-2'
%layer_normalization_2/batchnorm/Rsqrtа
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOpо
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2%
#layer_normalization_2/batchnorm/mulљ
%layer_normalization_2/batchnorm/mul_1Muladd:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_2/batchnorm/mul_1б
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_2/batchnorm/mul_2‘
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_2/batchnorm/ReadVariableOpк
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2%
#layer_normalization_2/batchnorm/subб
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_2/batchnorm/add_1’
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_1/dense_2/Tensordot/ReadVariableOpФ
#sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_2/Tensordot/axesЫ
#sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_2/Tensordot/free•
$sequential_1/dense_2/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/ShapeЮ
,sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/GatherV2/axisЇ
'sequential_1/dense_2/Tensordot/GatherV2GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/free:output:05sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/GatherV2Ґ
.sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_2/Tensordot/GatherV2_1/axisј
)sequential_1/dense_2/Tensordot/GatherV2_1GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/axes:output:07sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_2/Tensordot/GatherV2_1Ц
$sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_2/Tensordot/Const‘
#sequential_1/dense_2/Tensordot/ProdProd0sequential_1/dense_2/Tensordot/GatherV2:output:0-sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_2/Tensordot/ProdЪ
&sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_1№
%sequential_1/dense_2/Tensordot/Prod_1Prod2sequential_1/dense_2/Tensordot/GatherV2_1:output:0/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_2/Tensordot/Prod_1Ъ
*sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_2/Tensordot/concat/axisЩ
%sequential_1/dense_2/Tensordot/concatConcatV2,sequential_1/dense_2/Tensordot/free:output:0,sequential_1/dense_2/Tensordot/axes:output:03sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_2/Tensordot/concatа
$sequential_1/dense_2/Tensordot/stackPack,sequential_1/dense_2/Tensordot/Prod:output:0.sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/stackт
(sequential_1/dense_2/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0.sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2*
(sequential_1/dense_2/Tensordot/transposeу
&sequential_1/dense_2/Tensordot/ReshapeReshape,sequential_1/dense_2/Tensordot/transpose:y:0-sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_1/dense_2/Tensordot/Reshapeт
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_1/dense_2/Tensordot/MatMulЪ
&sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_2Ю
,sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/concat_1/axis¶
'sequential_1/dense_2/Tensordot/concat_1ConcatV20sequential_1/dense_2/Tensordot/GatherV2:output:0/sequential_1/dense_2/Tensordot/Const_2:output:05sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/concat_1д
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:00sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2 
sequential_1/dense_2/TensordotЋ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOpџ
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
sequential_1/dense_2/BiasAddЫ
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
sequential_1/dense_2/Relu’
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_1/dense_3/Tensordot/ReadVariableOpФ
#sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_3/Tensordot/axesЫ
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
$sequential_1/dense_3/Tensordot/ShapeЮ
,sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/GatherV2/axisЇ
'sequential_1/dense_3/Tensordot/GatherV2GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/free:output:05sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/GatherV2Ґ
.sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_3/Tensordot/GatherV2_1/axisј
)sequential_1/dense_3/Tensordot/GatherV2_1GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/axes:output:07sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_3/Tensordot/GatherV2_1Ц
$sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_3/Tensordot/Const‘
#sequential_1/dense_3/Tensordot/ProdProd0sequential_1/dense_3/Tensordot/GatherV2:output:0-sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_3/Tensordot/ProdЪ
&sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_1№
%sequential_1/dense_3/Tensordot/Prod_1Prod2sequential_1/dense_3/Tensordot/GatherV2_1:output:0/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_3/Tensordot/Prod_1Ъ
*sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_3/Tensordot/concat/axisЩ
%sequential_1/dense_3/Tensordot/concatConcatV2,sequential_1/dense_3/Tensordot/free:output:0,sequential_1/dense_3/Tensordot/axes:output:03sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_3/Tensordot/concatа
$sequential_1/dense_3/Tensordot/stackPack,sequential_1/dense_3/Tensordot/Prod:output:0.sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/stackр
(sequential_1/dense_3/Tensordot/transpose	Transpose'sequential_1/dense_2/Relu:activations:0.sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2*
(sequential_1/dense_3/Tensordot/transposeу
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_3/Tensordot/transpose:y:0-sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_1/dense_3/Tensordot/Reshapeт
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_1/dense_3/Tensordot/MatMulЪ
&sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_2Ю
,sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/concat_1/axis¶
'sequential_1/dense_3/Tensordot/concat_1ConcatV20sequential_1/dense_3/Tensordot/GatherV2:output:0/sequential_1/dense_3/Tensordot/Const_2:output:05sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/concat_1д
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:00sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2 
sequential_1/dense_3/TensordotЋ
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOpџ
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
sequential_1/dense_3/BiasAddw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_3/dropout/Constі
dropout_3/dropout/MulMul%sequential_1/dense_3/BiasAdd:output:0 dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dropout_3/dropout/MulЗ
dropout_3/dropout/ShapeShape%sequential_1/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shapeп
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€- *
dtype0*

seed**
seed220
.dropout_3/dropout/random_uniform/RandomUniformЙ
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_3/dropout/GreaterEqual/yк
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2 
dropout_3/dropout/GreaterEqual°
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€- 2
dropout_3/dropout/Cast¶
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dropout_3/dropout/Mul_1Х
add_1AddV2)layer_normalization_2/batchnorm/add_1:z:0dropout_3/dropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
add_1ґ
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indicesб
"layer_normalization_3/moments/meanMean	add_1:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2$
"layer_normalization_3/moments/meanЋ
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2,
*layer_normalization_3/moments/StopGradientн
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 21
/layer_normalization_3/moments/SquaredDifferenceЊ
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indicesЧ
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2(
&layer_normalization_3/moments/varianceУ
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_3/batchnorm/add/yк
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2%
#layer_normalization_3/batchnorm/addґ
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€-2'
%layer_normalization_3/batchnorm/Rsqrtа
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOpо
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2%
#layer_normalization_3/batchnorm/mulњ
%layer_normalization_3/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_3/batchnorm/mul_1б
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_3/batchnorm/mul_2‘
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_3/batchnorm/ReadVariableOpк
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2%
#layer_normalization_3/batchnorm/subб
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_3/batchnorm/add_1”
IdentityIdentity)layer_normalization_3/batchnorm/add_1:z:0/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€- ::::::::::::::::2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2М
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
:€€€€€€€€€- 
 
_user_specified_nameinputs
М
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_89042

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeј
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seed*2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ѕ
б
B__inference_dense_3_layer_call_and_return_conditional_losses_90686

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackР
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Р
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЗ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2	
BiasAddЬ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€- ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€- 
 
_user_specified_nameinputs
џ
э
G__inference_sequential_1_layer_call_and_return_conditional_losses_88400
dense_2_input
dense_2_88348
dense_2_88350
dense_3_88394
dense_3_88396
identityИҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallЪ
dense_2/StatefulPartitionedCallStatefulPartitionedCalldense_2_inputdense_2_88348dense_2_88350*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_883372!
dense_2/StatefulPartitionedCallµ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_88394dense_3_88396*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_883832!
dense_3/StatefulPartitionedCallƒ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€- ::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:Z V
+
_output_shapes
:€€€€€€€€€- 
'
_user_specified_namedense_2_input
П
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
:€€€€€€€€€€€€€€€€€€2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ѕ
ы
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_88517
x&
"embedding_1_embedding_lookup_88504$
 embedding_embedding_lookup_88510
identityИҐembedding/embedding_lookupҐembedding_1/embedding_lookup?
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
€€€€€€€€€2
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
strided_slice/stack_2в
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
range/deltaА
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:€€€€€€€€€2
range≠
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_88504range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/88504*'
_output_shapes
:€€€€€€€€€ *
dtype02
embedding_1/embedding_lookupШ
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/88504*'
_output_shapes
:€€€€€€€€€ 2'
%embedding_1/embedding_lookup/Identityј
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'embedding_1/embedding_lookup/Identity_1m
embedding/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€Љi2
embedding/CastЃ
embedding/embedding_lookupResourceGather embedding_embedding_lookup_88510embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/88510*,
_output_shapes
:€€€€€€€€€Љi *
dtype02
embedding/embedding_lookupХ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/88510*,
_output_shapes
:€€€€€€€€€Љi 2%
#embedding/embedding_lookup/Identityњ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€Љi 2'
%embedding/embedding_lookup/Identity_1ђ
addAddV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Љi 2
addЬ
IdentityIdentityadd:z:0^embedding/embedding_lookup^embedding_1/embedding_lookup*
T0*,
_output_shapes
:€€€€€€€€€Љi 2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€Љi::28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:K G
(
_output_shapes
:€€€€€€€€€Љi

_user_specified_namex
«
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_90400

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
∆
ц
G__inference_sequential_1_layer_call_and_return_conditional_losses_88431

inputs
dense_2_88420
dense_2_88422
dense_3_88425
dense_3_88427
identityИҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallУ
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_88420dense_2_88422*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_883372!
dense_2/StatefulPartitionedCallµ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_88425dense_3_88427*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_883832!
dense_3/StatefulPartitionedCallƒ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€- ::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€- 
 
_user_specified_nameinputs
л
|
'__inference_dense_3_layer_call_fn_90695

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_883832
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€- ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€- 
 
_user_specified_nameinputs
џ
э
G__inference_sequential_1_layer_call_and_return_conditional_losses_88414
dense_2_input
dense_2_88403
dense_2_88405
dense_3_88408
dense_3_88410
identityИҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallЪ
dense_2/StatefulPartitionedCallStatefulPartitionedCalldense_2_inputdense_2_88403dense_2_88405*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_883372!
dense_2/StatefulPartitionedCallµ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_88408dense_3_88410*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_883832!
dense_3/StatefulPartitionedCallƒ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€- ::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:Z V
+
_output_shapes
:€€€€€€€€€- 
'
_user_specified_namedense_2_input
ґ
я
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
identityИҐStatefulPartitionedCall©
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
:€€€€€€€€€*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_892172
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ь
_input_shapesК
З:€€€€€€€€€Љi:€€€€€€€€€::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€Љi
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_2
Љ
б
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
identityИҐStatefulPartitionedCallЂ
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
:€€€€€€€€€*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_892172
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ь
_input_shapesК
З:€€€€€€€€€Љi:€€€€€€€€€::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:€€€€€€€€€Љi
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
мь
—
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
identityИҐ.layer_normalization_2/batchnorm/ReadVariableOpҐ2layer_normalization_2/batchnorm/mul/ReadVariableOpҐ.layer_normalization_3/batchnorm/ReadVariableOpҐ2layer_normalization_3/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_1/attention_output/add/ReadVariableOpҐDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_1/key/add/ReadVariableOpҐ7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_1/query/add/ReadVariableOpҐ9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_1/value/add/ReadVariableOpҐ9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpҐ+sequential_1/dense_2/BiasAdd/ReadVariableOpҐ-sequential_1/dense_2/Tensordot/ReadVariableOpҐ+sequential_1/dense_3/BiasAdd/ReadVariableOpҐ-sequential_1/dense_3/Tensordot/ReadVariableOpэ
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpН
*multi_head_attention_1/query/einsum/EinsumEinsuminputsAmulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/Einsumџ
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/query/add/ReadVariableOpх
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 2"
 multi_head_attention_1/query/addч
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpЗ
(multi_head_attention_1/key/einsum/EinsumEinsuminputs?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/Einsum’
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_1/key/add/ReadVariableOpн
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 2 
multi_head_attention_1/key/addэ
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpН
*multi_head_attention_1/value/einsum/EinsumEinsuminputsAmulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/Einsumџ
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/value/add/ReadVariableOpх
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 2"
 multi_head_attention_1/value/addБ
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2
multi_head_attention_1/Mul/y∆
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€- 2
multi_head_attention_1/Mulь
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€--*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/Einsumƒ
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€--2(
&multi_head_attention_1/softmax/Softmax°
,multi_head_attention_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,multi_head_attention_1/dropout/dropout/ConstВ
*multi_head_attention_1/dropout/dropout/MulMul0multi_head_attention_1/softmax/Softmax:softmax:05multi_head_attention_1/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€--2,
*multi_head_attention_1/dropout/dropout/MulЉ
,multi_head_attention_1/dropout/dropout/ShapeShape0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_1/dropout/dropout/Shape•
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_1/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€--*
dtype0*

seed*2E
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform≥
5multi_head_attention_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_1/dropout/dropout/GreaterEqual/y¬
3multi_head_attention_1/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_1/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€--25
3multi_head_attention_1/dropout/dropout/GreaterEqualд
+multi_head_attention_1/dropout/dropout/CastCast7multi_head_attention_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€--2-
+multi_head_attention_1/dropout/dropout/Castю
,multi_head_attention_1/dropout/dropout/Mul_1Mul.multi_head_attention_1/dropout/dropout/Mul:z:0/multi_head_attention_1/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€--2.
,multi_head_attention_1/dropout/dropout/Mul_1Ф
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/dropout/Mul_1:z:0$multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/EinsumЮ
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp”
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€- *
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/Einsumш
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOpЭ
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2-
+multi_head_attention_1/attention_output/addw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_2/dropout/ConstЊ
dropout_2/dropout/MulMul/multi_head_attention_1/attention_output/add:z:0 dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dropout_2/dropout/MulС
dropout_2/dropout/ShapeShape/multi_head_attention_1/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shapeп
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€- *
dtype0*

seed**
seed220
.dropout_2/dropout/random_uniform/RandomUniformЙ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_2/dropout/GreaterEqual/yк
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2 
dropout_2/dropout/GreaterEqual°
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€- 2
dropout_2/dropout/Cast¶
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dropout_2/dropout/Mul_1n
addAddV2inputsdropout_2/dropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
addґ
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indicesя
"layer_normalization_2/moments/meanMeanadd:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2$
"layer_normalization_2/moments/meanЋ
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2,
*layer_normalization_2/moments/StopGradientл
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 21
/layer_normalization_2/moments/SquaredDifferenceЊ
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indicesЧ
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2(
&layer_normalization_2/moments/varianceУ
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_2/batchnorm/add/yк
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2%
#layer_normalization_2/batchnorm/addґ
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€-2'
%layer_normalization_2/batchnorm/Rsqrtа
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOpо
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2%
#layer_normalization_2/batchnorm/mulљ
%layer_normalization_2/batchnorm/mul_1Muladd:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_2/batchnorm/mul_1б
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_2/batchnorm/mul_2‘
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_2/batchnorm/ReadVariableOpк
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2%
#layer_normalization_2/batchnorm/subб
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_2/batchnorm/add_1’
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_1/dense_2/Tensordot/ReadVariableOpФ
#sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_2/Tensordot/axesЫ
#sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_2/Tensordot/free•
$sequential_1/dense_2/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/ShapeЮ
,sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/GatherV2/axisЇ
'sequential_1/dense_2/Tensordot/GatherV2GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/free:output:05sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/GatherV2Ґ
.sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_2/Tensordot/GatherV2_1/axisј
)sequential_1/dense_2/Tensordot/GatherV2_1GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/axes:output:07sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_2/Tensordot/GatherV2_1Ц
$sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_2/Tensordot/Const‘
#sequential_1/dense_2/Tensordot/ProdProd0sequential_1/dense_2/Tensordot/GatherV2:output:0-sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_2/Tensordot/ProdЪ
&sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_1№
%sequential_1/dense_2/Tensordot/Prod_1Prod2sequential_1/dense_2/Tensordot/GatherV2_1:output:0/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_2/Tensordot/Prod_1Ъ
*sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_2/Tensordot/concat/axisЩ
%sequential_1/dense_2/Tensordot/concatConcatV2,sequential_1/dense_2/Tensordot/free:output:0,sequential_1/dense_2/Tensordot/axes:output:03sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_2/Tensordot/concatа
$sequential_1/dense_2/Tensordot/stackPack,sequential_1/dense_2/Tensordot/Prod:output:0.sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/stackт
(sequential_1/dense_2/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0.sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2*
(sequential_1/dense_2/Tensordot/transposeу
&sequential_1/dense_2/Tensordot/ReshapeReshape,sequential_1/dense_2/Tensordot/transpose:y:0-sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_1/dense_2/Tensordot/Reshapeт
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_1/dense_2/Tensordot/MatMulЪ
&sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_2Ю
,sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/concat_1/axis¶
'sequential_1/dense_2/Tensordot/concat_1ConcatV20sequential_1/dense_2/Tensordot/GatherV2:output:0/sequential_1/dense_2/Tensordot/Const_2:output:05sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/concat_1д
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:00sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2 
sequential_1/dense_2/TensordotЋ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOpџ
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
sequential_1/dense_2/BiasAddЫ
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
sequential_1/dense_2/Relu’
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_1/dense_3/Tensordot/ReadVariableOpФ
#sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_3/Tensordot/axesЫ
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
$sequential_1/dense_3/Tensordot/ShapeЮ
,sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/GatherV2/axisЇ
'sequential_1/dense_3/Tensordot/GatherV2GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/free:output:05sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/GatherV2Ґ
.sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_3/Tensordot/GatherV2_1/axisј
)sequential_1/dense_3/Tensordot/GatherV2_1GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/axes:output:07sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_3/Tensordot/GatherV2_1Ц
$sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_3/Tensordot/Const‘
#sequential_1/dense_3/Tensordot/ProdProd0sequential_1/dense_3/Tensordot/GatherV2:output:0-sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_3/Tensordot/ProdЪ
&sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_1№
%sequential_1/dense_3/Tensordot/Prod_1Prod2sequential_1/dense_3/Tensordot/GatherV2_1:output:0/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_3/Tensordot/Prod_1Ъ
*sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_3/Tensordot/concat/axisЩ
%sequential_1/dense_3/Tensordot/concatConcatV2,sequential_1/dense_3/Tensordot/free:output:0,sequential_1/dense_3/Tensordot/axes:output:03sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_3/Tensordot/concatа
$sequential_1/dense_3/Tensordot/stackPack,sequential_1/dense_3/Tensordot/Prod:output:0.sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/stackр
(sequential_1/dense_3/Tensordot/transpose	Transpose'sequential_1/dense_2/Relu:activations:0.sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2*
(sequential_1/dense_3/Tensordot/transposeу
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_3/Tensordot/transpose:y:0-sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_1/dense_3/Tensordot/Reshapeт
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_1/dense_3/Tensordot/MatMulЪ
&sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_2Ю
,sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/concat_1/axis¶
'sequential_1/dense_3/Tensordot/concat_1ConcatV20sequential_1/dense_3/Tensordot/GatherV2:output:0/sequential_1/dense_3/Tensordot/Const_2:output:05sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/concat_1д
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:00sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2 
sequential_1/dense_3/TensordotЋ
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOpџ
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
sequential_1/dense_3/BiasAddw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_3/dropout/Constі
dropout_3/dropout/MulMul%sequential_1/dense_3/BiasAdd:output:0 dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dropout_3/dropout/MulЗ
dropout_3/dropout/ShapeShape%sequential_1/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shapeп
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€- *
dtype0*

seed**
seed220
.dropout_3/dropout/random_uniform/RandomUniformЙ
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_3/dropout/GreaterEqual/yк
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2 
dropout_3/dropout/GreaterEqual°
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€- 2
dropout_3/dropout/Cast¶
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dropout_3/dropout/Mul_1Х
add_1AddV2)layer_normalization_2/batchnorm/add_1:z:0dropout_3/dropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
add_1ґ
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indicesб
"layer_normalization_3/moments/meanMean	add_1:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2$
"layer_normalization_3/moments/meanЋ
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2,
*layer_normalization_3/moments/StopGradientн
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 21
/layer_normalization_3/moments/SquaredDifferenceЊ
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indicesЧ
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2(
&layer_normalization_3/moments/varianceУ
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_3/batchnorm/add/yк
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2%
#layer_normalization_3/batchnorm/addґ
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€-2'
%layer_normalization_3/batchnorm/Rsqrtа
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOpо
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2%
#layer_normalization_3/batchnorm/mulњ
%layer_normalization_3/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_3/batchnorm/mul_1б
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_3/batchnorm/mul_2‘
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_3/batchnorm/ReadVariableOpк
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2%
#layer_normalization_3/batchnorm/subб
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_3/batchnorm/add_1”
IdentityIdentity)layer_normalization_3/batchnorm/add_1:z:0/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€- ::::::::::::::::2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2М
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
:€€€€€€€€€- 
 
_user_specified_nameinputs
ў
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
:€€€€€€€€€ 2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€- :S O
+
_output_shapes
:€€€€€€€€€- 
 
_user_specified_nameinputs
∆
ц
G__inference_sequential_1_layer_call_and_return_conditional_losses_88458

inputs
dense_2_88447
dense_2_88449
dense_3_88452
dense_3_88454
identityИҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallУ
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_88447dense_2_88449*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_883372!
dense_2/StatefulPartitionedCallµ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_88452dense_3_88454*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_883832!
dense_3/StatefulPartitionedCallƒ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€- ::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€- 
 
_user_specified_nameinputs
А№
—
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
identityИҐ.layer_normalization_2/batchnorm/ReadVariableOpҐ2layer_normalization_2/batchnorm/mul/ReadVariableOpҐ.layer_normalization_3/batchnorm/ReadVariableOpҐ2layer_normalization_3/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_1/attention_output/add/ReadVariableOpҐDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_1/key/add/ReadVariableOpҐ7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_1/query/add/ReadVariableOpҐ9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_1/value/add/ReadVariableOpҐ9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpҐ+sequential_1/dense_2/BiasAdd/ReadVariableOpҐ-sequential_1/dense_2/Tensordot/ReadVariableOpҐ+sequential_1/dense_3/BiasAdd/ReadVariableOpҐ-sequential_1/dense_3/Tensordot/ReadVariableOpэ
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpН
*multi_head_attention_1/query/einsum/EinsumEinsuminputsAmulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/Einsumџ
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/query/add/ReadVariableOpх
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 2"
 multi_head_attention_1/query/addч
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpЗ
(multi_head_attention_1/key/einsum/EinsumEinsuminputs?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/Einsum’
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_1/key/add/ReadVariableOpн
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 2 
multi_head_attention_1/key/addэ
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpН
*multi_head_attention_1/value/einsum/EinsumEinsuminputsAmulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/Einsumџ
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_1/value/add/ReadVariableOpх
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 2"
 multi_head_attention_1/value/addБ
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2
multi_head_attention_1/Mul/y∆
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€- 2
multi_head_attention_1/Mulь
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€--*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/Einsumƒ
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€--2(
&multi_head_attention_1/softmax/Softmax 
'multi_head_attention_1/dropout/IdentityIdentity0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€--2)
'multi_head_attention_1/dropout/IdentityФ
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/Identity:output:0$multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/EinsumЮ
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp”
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€- *
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/Einsumш
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOpЭ
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2-
+multi_head_attention_1/attention_output/addЫ
dropout_2/IdentityIdentity/multi_head_attention_1/attention_output/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dropout_2/Identityn
addAddV2inputsdropout_2/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
addґ
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indicesя
"layer_normalization_2/moments/meanMeanadd:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2$
"layer_normalization_2/moments/meanЋ
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2,
*layer_normalization_2/moments/StopGradientл
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 21
/layer_normalization_2/moments/SquaredDifferenceЊ
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indicesЧ
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2(
&layer_normalization_2/moments/varianceУ
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_2/batchnorm/add/yк
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2%
#layer_normalization_2/batchnorm/addґ
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€-2'
%layer_normalization_2/batchnorm/Rsqrtа
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOpо
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2%
#layer_normalization_2/batchnorm/mulљ
%layer_normalization_2/batchnorm/mul_1Muladd:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_2/batchnorm/mul_1б
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_2/batchnorm/mul_2‘
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_2/batchnorm/ReadVariableOpк
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2%
#layer_normalization_2/batchnorm/subб
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_2/batchnorm/add_1’
-sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_1/dense_2/Tensordot/ReadVariableOpФ
#sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_2/Tensordot/axesЫ
#sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_1/dense_2/Tensordot/free•
$sequential_1/dense_2/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/ShapeЮ
,sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/GatherV2/axisЇ
'sequential_1/dense_2/Tensordot/GatherV2GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/free:output:05sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/GatherV2Ґ
.sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_2/Tensordot/GatherV2_1/axisј
)sequential_1/dense_2/Tensordot/GatherV2_1GatherV2-sequential_1/dense_2/Tensordot/Shape:output:0,sequential_1/dense_2/Tensordot/axes:output:07sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_2/Tensordot/GatherV2_1Ц
$sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_2/Tensordot/Const‘
#sequential_1/dense_2/Tensordot/ProdProd0sequential_1/dense_2/Tensordot/GatherV2:output:0-sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_2/Tensordot/ProdЪ
&sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_1№
%sequential_1/dense_2/Tensordot/Prod_1Prod2sequential_1/dense_2/Tensordot/GatherV2_1:output:0/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_2/Tensordot/Prod_1Ъ
*sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_2/Tensordot/concat/axisЩ
%sequential_1/dense_2/Tensordot/concatConcatV2,sequential_1/dense_2/Tensordot/free:output:0,sequential_1/dense_2/Tensordot/axes:output:03sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_2/Tensordot/concatа
$sequential_1/dense_2/Tensordot/stackPack,sequential_1/dense_2/Tensordot/Prod:output:0.sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_2/Tensordot/stackт
(sequential_1/dense_2/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0.sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2*
(sequential_1/dense_2/Tensordot/transposeу
&sequential_1/dense_2/Tensordot/ReshapeReshape,sequential_1/dense_2/Tensordot/transpose:y:0-sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_1/dense_2/Tensordot/Reshapeт
%sequential_1/dense_2/Tensordot/MatMulMatMul/sequential_1/dense_2/Tensordot/Reshape:output:05sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_1/dense_2/Tensordot/MatMulЪ
&sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_2/Tensordot/Const_2Ю
,sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_2/Tensordot/concat_1/axis¶
'sequential_1/dense_2/Tensordot/concat_1ConcatV20sequential_1/dense_2/Tensordot/GatherV2:output:0/sequential_1/dense_2/Tensordot/Const_2:output:05sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_2/Tensordot/concat_1д
sequential_1/dense_2/TensordotReshape/sequential_1/dense_2/Tensordot/MatMul:product:00sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2 
sequential_1/dense_2/TensordotЋ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_2/BiasAdd/ReadVariableOpџ
sequential_1/dense_2/BiasAddBiasAdd'sequential_1/dense_2/Tensordot:output:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
sequential_1/dense_2/BiasAddЫ
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
sequential_1/dense_2/Relu’
-sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02/
-sequential_1/dense_3/Tensordot/ReadVariableOpФ
#sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_3/Tensordot/axesЫ
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
$sequential_1/dense_3/Tensordot/ShapeЮ
,sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/GatherV2/axisЇ
'sequential_1/dense_3/Tensordot/GatherV2GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/free:output:05sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/GatherV2Ґ
.sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_3/Tensordot/GatherV2_1/axisј
)sequential_1/dense_3/Tensordot/GatherV2_1GatherV2-sequential_1/dense_3/Tensordot/Shape:output:0,sequential_1/dense_3/Tensordot/axes:output:07sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_3/Tensordot/GatherV2_1Ц
$sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_3/Tensordot/Const‘
#sequential_1/dense_3/Tensordot/ProdProd0sequential_1/dense_3/Tensordot/GatherV2:output:0-sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_3/Tensordot/ProdЪ
&sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_1№
%sequential_1/dense_3/Tensordot/Prod_1Prod2sequential_1/dense_3/Tensordot/GatherV2_1:output:0/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_3/Tensordot/Prod_1Ъ
*sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_3/Tensordot/concat/axisЩ
%sequential_1/dense_3/Tensordot/concatConcatV2,sequential_1/dense_3/Tensordot/free:output:0,sequential_1/dense_3/Tensordot/axes:output:03sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_3/Tensordot/concatа
$sequential_1/dense_3/Tensordot/stackPack,sequential_1/dense_3/Tensordot/Prod:output:0.sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_3/Tensordot/stackр
(sequential_1/dense_3/Tensordot/transpose	Transpose'sequential_1/dense_2/Relu:activations:0.sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2*
(sequential_1/dense_3/Tensordot/transposeу
&sequential_1/dense_3/Tensordot/ReshapeReshape,sequential_1/dense_3/Tensordot/transpose:y:0-sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_1/dense_3/Tensordot/Reshapeт
%sequential_1/dense_3/Tensordot/MatMulMatMul/sequential_1/dense_3/Tensordot/Reshape:output:05sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_1/dense_3/Tensordot/MatMulЪ
&sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_3/Tensordot/Const_2Ю
,sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_3/Tensordot/concat_1/axis¶
'sequential_1/dense_3/Tensordot/concat_1ConcatV20sequential_1/dense_3/Tensordot/GatherV2:output:0/sequential_1/dense_3/Tensordot/Const_2:output:05sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_3/Tensordot/concat_1д
sequential_1/dense_3/TensordotReshape/sequential_1/dense_3/Tensordot/MatMul:product:00sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2 
sequential_1/dense_3/TensordotЋ
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_1/dense_3/BiasAdd/ReadVariableOpџ
sequential_1/dense_3/BiasAddBiasAdd'sequential_1/dense_3/Tensordot:output:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
sequential_1/dense_3/BiasAddС
dropout_3/IdentityIdentity%sequential_1/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dropout_3/IdentityХ
add_1AddV2)layer_normalization_2/batchnorm/add_1:z:0dropout_3/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
add_1ґ
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indicesб
"layer_normalization_3/moments/meanMean	add_1:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2$
"layer_normalization_3/moments/meanЋ
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2,
*layer_normalization_3/moments/StopGradientн
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 21
/layer_normalization_3/moments/SquaredDifferenceЊ
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indicesЧ
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2(
&layer_normalization_3/moments/varianceУ
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_3/batchnorm/add/yк
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2%
#layer_normalization_3/batchnorm/addґ
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€-2'
%layer_normalization_3/batchnorm/Rsqrtа
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOpо
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2%
#layer_normalization_3/batchnorm/mulњ
%layer_normalization_3/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_3/batchnorm/mul_1б
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_3/batchnorm/mul_2‘
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_3/batchnorm/ReadVariableOpк
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2%
#layer_normalization_3/batchnorm/subб
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2'
%layer_normalization_3/batchnorm/add_1”
IdentityIdentity)layer_normalization_3/batchnorm/add_1:z:0/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp.^sequential_1/dense_2/Tensordot/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp.^sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€- ::::::::::::::::2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2М
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
:€€€€€€€€€- 
 
_user_specified_nameinputs
ф
h
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_88296

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

ExpandDimsЉ
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
ђ*
paddingVALID*
strides	
ђ2	
AvgPoolО
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
°
b
)__inference_dropout_5_layer_call_fn_90452

inputs
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_890422
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Х
E
)__inference_dropout_5_layer_call_fn_90457

inputs
identity≈
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_890472
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
а}
Й
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

identity_1ИҐMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameї
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*Ќ
value√Bј7B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/8/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesч
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*Б
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesґ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableopLsavev2_token_and_position_embedding_embedding_embeddings_read_readvariableopNsavev2_token_and_position_embedding_embedding_1_embeddings_read_readvariableopRsavev2_transformer_block_1_multi_head_attention_1_query_kernel_read_readvariableopPsavev2_transformer_block_1_multi_head_attention_1_query_bias_read_readvariableopPsavev2_transformer_block_1_multi_head_attention_1_key_kernel_read_readvariableopNsavev2_transformer_block_1_multi_head_attention_1_key_bias_read_readvariableopRsavev2_transformer_block_1_multi_head_attention_1_value_kernel_read_readvariableopPsavev2_transformer_block_1_multi_head_attention_1_value_bias_read_readvariableop]savev2_transformer_block_1_multi_head_attention_1_attention_output_kernel_read_readvariableop[savev2_transformer_block_1_multi_head_attention_1_attention_output_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableopJsavev2_transformer_block_1_layer_normalization_2_gamma_read_readvariableopIsavev2_transformer_block_1_layer_normalization_2_beta_read_readvariableopJsavev2_transformer_block_1_layer_normalization_3_gamma_read_readvariableopIsavev2_transformer_block_1_layer_normalization_3_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_sgd_dense_4_kernel_momentum_read_readvariableop4savev2_sgd_dense_4_bias_momentum_read_readvariableop6savev2_sgd_dense_5_kernel_momentum_read_readvariableop4savev2_sgd_dense_5_bias_momentum_read_readvariableop6savev2_sgd_dense_6_kernel_momentum_read_readvariableop4savev2_sgd_dense_6_bias_momentum_read_readvariableopYsavev2_sgd_token_and_position_embedding_embedding_embeddings_momentum_read_readvariableop[savev2_sgd_token_and_position_embedding_embedding_1_embeddings_momentum_read_readvariableop_savev2_sgd_transformer_block_1_multi_head_attention_1_query_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_1_multi_head_attention_1_query_bias_momentum_read_readvariableop]savev2_sgd_transformer_block_1_multi_head_attention_1_key_kernel_momentum_read_readvariableop[savev2_sgd_transformer_block_1_multi_head_attention_1_key_bias_momentum_read_readvariableop_savev2_sgd_transformer_block_1_multi_head_attention_1_value_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_1_multi_head_attention_1_value_bias_momentum_read_readvariableopjsavev2_sgd_transformer_block_1_multi_head_attention_1_attention_output_kernel_momentum_read_readvariableophsavev2_sgd_transformer_block_1_multi_head_attention_1_attention_output_bias_momentum_read_readvariableop6savev2_sgd_dense_2_kernel_momentum_read_readvariableop4savev2_sgd_dense_2_bias_momentum_read_readvariableop6savev2_sgd_dense_3_kernel_momentum_read_readvariableop4savev2_sgd_dense_3_bias_momentum_read_readvariableopWsavev2_sgd_transformer_block_1_layer_normalization_2_gamma_momentum_read_readvariableopVsavev2_sgd_transformer_block_1_layer_normalization_2_beta_momentum_read_readvariableopWsavev2_sgd_transformer_block_1_layer_normalization_3_gamma_momentum_read_readvariableopVsavev2_sgd_transformer_block_1_layer_normalization_3_beta_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *E
dtypes;
927	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*„
_input_shapes≈
¬: :( : :  : : :: : : : : :	Љi :  : :  : :  : :  : :  : :  : : : : : : : :( : :  : : :: :	Љi :  : :  : :  : :  : :  : :  : : : : : : 2(
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
:	Љi :($
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
:	Љi :('$
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
ґ
Я
,__inference_sequential_1_layer_call_fn_90616

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_884582
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€- ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€- 
 
_user_specified_nameinputs
ЇМ
Э
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
identityИҐ$model/dense_4/BiasAdd/ReadVariableOpҐ#model/dense_4/MatMul/ReadVariableOpҐ$model/dense_5/BiasAdd/ReadVariableOpҐ#model/dense_5/MatMul/ReadVariableOpҐ$model/dense_6/BiasAdd/ReadVariableOpҐ#model/dense_6/MatMul/ReadVariableOpҐ=model/token_and_position_embedding/embedding/embedding_lookupҐ?model/token_and_position_embedding/embedding_1/embedding_lookupҐHmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpҐLmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpҐHmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpҐLmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpҐTmodel/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpҐ^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpҐGmodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOpҐQmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpҐImodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpҐSmodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpҐImodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpҐSmodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpҐEmodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpҐGmodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpҐEmodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpҐGmodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpЛ
(model/token_and_position_embedding/ShapeShapeinput_1*
T0*
_output_shapes
:2*
(model/token_and_position_embedding/Shape√
6model/token_and_position_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€28
6model/token_and_position_embedding/strided_slice/stackЊ
8model/token_and_position_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8model/token_and_position_embedding/strided_slice/stack_1Њ
8model/token_and_position_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model/token_and_position_embedding/strided_slice/stack_2і
0model/token_and_position_embedding/strided_sliceStridedSlice1model/token_and_position_embedding/Shape:output:0?model/token_and_position_embedding/strided_slice/stack:output:0Amodel/token_and_position_embedding/strided_slice/stack_1:output:0Amodel/token_and_position_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0model/token_and_position_embedding/strided_sliceҐ
.model/token_and_position_embedding/range/startConst*
_output_shapes
: *
dtype0*
value	B : 20
.model/token_and_position_embedding/range/startҐ
.model/token_and_position_embedding/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :20
.model/token_and_position_embedding/range/deltaѓ
(model/token_and_position_embedding/rangeRange7model/token_and_position_embedding/range/start:output:09model/token_and_position_embedding/strided_slice:output:07model/token_and_position_embedding/range/delta:output:0*#
_output_shapes
:€€€€€€€€€2*
(model/token_and_position_embedding/range№
?model/token_and_position_embedding/embedding_1/embedding_lookupResourceGatherEmodel_token_and_position_embedding_embedding_1_embedding_lookup_881211model/token_and_position_embedding/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*X
_classN
LJloc:@model/token_and_position_embedding/embedding_1/embedding_lookup/88121*'
_output_shapes
:€€€€€€€€€ *
dtype02A
?model/token_and_position_embedding/embedding_1/embedding_lookup§
Hmodel/token_and_position_embedding/embedding_1/embedding_lookup/IdentityIdentityHmodel/token_and_position_embedding/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@model/token_and_position_embedding/embedding_1/embedding_lookup/88121*'
_output_shapes
:€€€€€€€€€ 2J
Hmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity©
Jmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity_1IdentityQmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2L
Jmodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity_1є
1model/token_and_position_embedding/embedding/CastCastinput_1*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€Љi23
1model/token_and_position_embedding/embedding/CastЁ
=model/token_and_position_embedding/embedding/embedding_lookupResourceGatherCmodel_token_and_position_embedding_embedding_embedding_lookup_881275model/token_and_position_embedding/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*V
_classL
JHloc:@model/token_and_position_embedding/embedding/embedding_lookup/88127*,
_output_shapes
:€€€€€€€€€Љi *
dtype02?
=model/token_and_position_embedding/embedding/embedding_lookup°
Fmodel/token_and_position_embedding/embedding/embedding_lookup/IdentityIdentityFmodel/token_and_position_embedding/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@model/token_and_position_embedding/embedding/embedding_lookup/88127*,
_output_shapes
:€€€€€€€€€Љi 2H
Fmodel/token_and_position_embedding/embedding/embedding_lookup/Identity®
Hmodel/token_and_position_embedding/embedding/embedding_lookup/Identity_1IdentityOmodel/token_and_position_embedding/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:€€€€€€€€€Љi 2J
Hmodel/token_and_position_embedding/embedding/embedding_lookup/Identity_1Є
&model/token_and_position_embedding/addAddV2Qmodel/token_and_position_embedding/embedding/embedding_lookup/Identity_1:output:0Smodel/token_and_position_embedding/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Љi 2(
&model/token_and_position_embedding/addТ
&model/average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&model/average_pooling1d/ExpandDims/dimо
"model/average_pooling1d/ExpandDims
ExpandDims*model/token_and_position_embedding/add:z:0/model/average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Љi 2$
"model/average_pooling1d/ExpandDimsт
model/average_pooling1d/AvgPoolAvgPool+model/average_pooling1d/ExpandDims:output:0*
T0*/
_output_shapes
:€€€€€€€€€- *
ksize	
ђ*
paddingVALID*
strides	
ђ2!
model/average_pooling1d/AvgPoolƒ
model/average_pooling1d/SqueezeSqueeze(model/average_pooling1d/AvgPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€- *
squeeze_dims
2!
model/average_pooling1d/SqueezeЋ
Smodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOp\model_transformer_block_1_multi_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02U
Smodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpэ
Dmodel/transformer_block_1/multi_head_attention_1/query/einsum/EinsumEinsum(model/average_pooling1d/Squeeze:output:0[model/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2F
Dmodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum©
Imodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpReadVariableOpRmodel_transformer_block_1_multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

: *
dtype02K
Imodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpЁ
:model/transformer_block_1/multi_head_attention_1/query/addAddV2Mmodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum:output:0Qmodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 2<
:model/transformer_block_1/multi_head_attention_1/query/add≈
Qmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOpZmodel_transformer_block_1_multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02S
Qmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpч
Bmodel/transformer_block_1/multi_head_attention_1/key/einsum/EinsumEinsum(model/average_pooling1d/Squeeze:output:0Ymodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2D
Bmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum£
Gmodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOpReadVariableOpPmodel_transformer_block_1_multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

: *
dtype02I
Gmodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOp’
8model/transformer_block_1/multi_head_attention_1/key/addAddV2Kmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum:output:0Omodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 2:
8model/transformer_block_1/multi_head_attention_1/key/addЋ
Smodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOp\model_transformer_block_1_multi_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02U
Smodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpэ
Dmodel/transformer_block_1/multi_head_attention_1/value/einsum/EinsumEinsum(model/average_pooling1d/Squeeze:output:0[model/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationabc,cde->abde2F
Dmodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum©
Imodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpReadVariableOpRmodel_transformer_block_1_multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

: *
dtype02K
Imodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpЁ
:model/transformer_block_1/multi_head_attention_1/value/addAddV2Mmodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum:output:0Qmodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€- 2<
:model/transformer_block_1/multi_head_attention_1/value/addµ
6model/transformer_block_1/multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>28
6model/transformer_block_1/multi_head_attention_1/Mul/yЃ
4model/transformer_block_1/multi_head_attention_1/MulMul>model/transformer_block_1/multi_head_attention_1/query/add:z:0?model/transformer_block_1/multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€- 26
4model/transformer_block_1/multi_head_attention_1/Mulд
>model/transformer_block_1/multi_head_attention_1/einsum/EinsumEinsum<model/transformer_block_1/multi_head_attention_1/key/add:z:08model/transformer_block_1/multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€--*
equationaecd,abcd->acbe2@
>model/transformer_block_1/multi_head_attention_1/einsum/EinsumТ
@model/transformer_block_1/multi_head_attention_1/softmax/SoftmaxSoftmaxGmodel/transformer_block_1/multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€--2B
@model/transformer_block_1/multi_head_attention_1/softmax/SoftmaxШ
Amodel/transformer_block_1/multi_head_attention_1/dropout/IdentityIdentityJmodel/transformer_block_1/multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€--2C
Amodel/transformer_block_1/multi_head_attention_1/dropout/Identityь
@model/transformer_block_1/multi_head_attention_1/einsum_1/EinsumEinsumJmodel/transformer_block_1/multi_head_attention_1/dropout/Identity:output:0>model/transformer_block_1/multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€- *
equationacbe,aecd->abcd2B
@model/transformer_block_1/multi_head_attention_1/einsum_1/Einsumм
^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpgmodel_transformer_block_1_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02`
^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpї
Omodel/transformer_block_1/multi_head_attention_1/attention_output/einsum/EinsumEinsumImodel/transformer_block_1/multi_head_attention_1/einsum_1/Einsum:output:0fmodel/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€- *
equationabcd,cde->abe2Q
Omodel/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum∆
Tmodel/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOp]model_transformer_block_1_multi_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02V
Tmodel/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpЕ
Emodel/transformer_block_1/multi_head_attention_1/attention_output/addAddV2Xmodel/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum:output:0\model/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2G
Emodel/transformer_block_1/multi_head_attention_1/attention_output/addй
,model/transformer_block_1/dropout_2/IdentityIdentityImodel/transformer_block_1/multi_head_attention_1/attention_output/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2.
,model/transformer_block_1/dropout_2/Identityё
model/transformer_block_1/addAddV2(model/average_pooling1d/Squeeze:output:05model/transformer_block_1/dropout_2/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
model/transformer_block_1/addк
Nmodel/transformer_block_1/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
Nmodel/transformer_block_1/layer_normalization_2/moments/mean/reduction_indices«
<model/transformer_block_1/layer_normalization_2/moments/meanMean!model/transformer_block_1/add:z:0Wmodel/transformer_block_1/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2>
<model/transformer_block_1/layer_normalization_2/moments/meanЩ
Dmodel/transformer_block_1/layer_normalization_2/moments/StopGradientStopGradientEmodel/transformer_block_1/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2F
Dmodel/transformer_block_1/layer_normalization_2/moments/StopGradient”
Imodel/transformer_block_1/layer_normalization_2/moments/SquaredDifferenceSquaredDifference!model/transformer_block_1/add:z:0Mmodel/transformer_block_1/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2K
Imodel/transformer_block_1/layer_normalization_2/moments/SquaredDifferenceт
Rmodel/transformer_block_1/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2T
Rmodel/transformer_block_1/layer_normalization_2/moments/variance/reduction_indices€
@model/transformer_block_1/layer_normalization_2/moments/varianceMeanMmodel/transformer_block_1/layer_normalization_2/moments/SquaredDifference:z:0[model/transformer_block_1/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2B
@model/transformer_block_1/layer_normalization_2/moments/variance«
?model/transformer_block_1/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52A
?model/transformer_block_1/layer_normalization_2/batchnorm/add/y“
=model/transformer_block_1/layer_normalization_2/batchnorm/addAddV2Imodel/transformer_block_1/layer_normalization_2/moments/variance:output:0Hmodel/transformer_block_1/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2?
=model/transformer_block_1/layer_normalization_2/batchnorm/addД
?model/transformer_block_1/layer_normalization_2/batchnorm/RsqrtRsqrtAmodel/transformer_block_1/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€-2A
?model/transformer_block_1/layer_normalization_2/batchnorm/RsqrtЃ
Lmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpUmodel_transformer_block_1_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02N
Lmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp÷
=model/transformer_block_1/layer_normalization_2/batchnorm/mulMulCmodel/transformer_block_1/layer_normalization_2/batchnorm/Rsqrt:y:0Tmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2?
=model/transformer_block_1/layer_normalization_2/batchnorm/mul•
?model/transformer_block_1/layer_normalization_2/batchnorm/mul_1Mul!model/transformer_block_1/add:z:0Amodel/transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2A
?model/transformer_block_1/layer_normalization_2/batchnorm/mul_1…
?model/transformer_block_1/layer_normalization_2/batchnorm/mul_2MulEmodel/transformer_block_1/layer_normalization_2/moments/mean:output:0Amodel/transformer_block_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2A
?model/transformer_block_1/layer_normalization_2/batchnorm/mul_2Ґ
Hmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpQmodel_transformer_block_1_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp“
=model/transformer_block_1/layer_normalization_2/batchnorm/subSubPmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp:value:0Cmodel/transformer_block_1/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2?
=model/transformer_block_1/layer_normalization_2/batchnorm/sub…
?model/transformer_block_1/layer_normalization_2/batchnorm/add_1AddV2Cmodel/transformer_block_1/layer_normalization_2/batchnorm/mul_1:z:0Amodel/transformer_block_1/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2A
?model/transformer_block_1/layer_normalization_2/batchnorm/add_1£
Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpReadVariableOpPmodel_transformer_block_1_sequential_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02I
Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp»
=model/transformer_block_1/sequential_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2?
=model/transformer_block_1/sequential_1/dense_2/Tensordot/axesѕ
=model/transformer_block_1/sequential_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model/transformer_block_1/sequential_1/dense_2/Tensordot/freeу
>model/transformer_block_1/sequential_1/dense_2/Tensordot/ShapeShapeCmodel/transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2@
>model/transformer_block_1/sequential_1/dense_2/Tensordot/Shape“
Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axisЉ
Amodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2GatherV2Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0Omodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Amodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2÷
Hmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axis¬
Cmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1GatherV2Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/Shape:output:0Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Qmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1 
>model/transformer_block_1/sequential_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2@
>model/transformer_block_1/sequential_1/dense_2/Tensordot/ConstЉ
=model/transformer_block_1/sequential_1/dense_2/Tensordot/ProdProdJmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2?
=model/transformer_block_1/sequential_1/dense_2/Tensordot/Prodќ
@model/transformer_block_1/sequential_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block_1/sequential_1/dense_2/Tensordot/Const_1ƒ
?model/transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1ProdLmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2_1:output:0Imodel/transformer_block_1/sequential_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2A
?model/transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1ќ
Dmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat/axisЫ
?model/transformer_block_1/sequential_1/dense_2/Tensordot/concatConcatV2Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/free:output:0Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/axes:output:0Mmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2A
?model/transformer_block_1/sequential_1/dense_2/Tensordot/concat»
>model/transformer_block_1/sequential_1/dense_2/Tensordot/stackPackFmodel/transformer_block_1/sequential_1/dense_2/Tensordot/Prod:output:0Hmodel/transformer_block_1/sequential_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2@
>model/transformer_block_1/sequential_1/dense_2/Tensordot/stackЏ
Bmodel/transformer_block_1/sequential_1/dense_2/Tensordot/transpose	TransposeCmodel/transformer_block_1/layer_normalization_2/batchnorm/add_1:z:0Hmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2D
Bmodel/transformer_block_1/sequential_1/dense_2/Tensordot/transposeџ
@model/transformer_block_1/sequential_1/dense_2/Tensordot/ReshapeReshapeFmodel/transformer_block_1/sequential_1/dense_2/Tensordot/transpose:y:0Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2B
@model/transformer_block_1/sequential_1/dense_2/Tensordot/ReshapeЏ
?model/transformer_block_1/sequential_1/dense_2/Tensordot/MatMulMatMulImodel/transformer_block_1/sequential_1/dense_2/Tensordot/Reshape:output:0Omodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2A
?model/transformer_block_1/sequential_1/dense_2/Tensordot/MatMulќ
@model/transformer_block_1/sequential_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block_1/sequential_1/dense_2/Tensordot/Const_2“
Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axis®
Amodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1ConcatV2Jmodel/transformer_block_1/sequential_1/dense_2/Tensordot/GatherV2:output:0Imodel/transformer_block_1/sequential_1/dense_2/Tensordot/Const_2:output:0Omodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1ћ
8model/transformer_block_1/sequential_1/dense_2/TensordotReshapeImodel/transformer_block_1/sequential_1/dense_2/Tensordot/MatMul:product:0Jmodel/transformer_block_1/sequential_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2:
8model/transformer_block_1/sequential_1/dense_2/TensordotЩ
Emodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOpNmodel_transformer_block_1_sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02G
Emodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp√
6model/transformer_block_1/sequential_1/dense_2/BiasAddBiasAddAmodel/transformer_block_1/sequential_1/dense_2/Tensordot:output:0Mmodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 28
6model/transformer_block_1/sequential_1/dense_2/BiasAddй
3model/transformer_block_1/sequential_1/dense_2/ReluRelu?model/transformer_block_1/sequential_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 25
3model/transformer_block_1/sequential_1/dense_2/Relu£
Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpReadVariableOpPmodel_transformer_block_1_sequential_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02I
Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp»
=model/transformer_block_1/sequential_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2?
=model/transformer_block_1/sequential_1/dense_3/Tensordot/axesѕ
=model/transformer_block_1/sequential_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model/transformer_block_1/sequential_1/dense_3/Tensordot/freeс
>model/transformer_block_1/sequential_1/dense_3/Tensordot/ShapeShapeAmodel/transformer_block_1/sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:2@
>model/transformer_block_1/sequential_1/dense_3/Tensordot/Shape“
Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axisЉ
Amodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2GatherV2Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0Omodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Amodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2÷
Hmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axis¬
Cmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1GatherV2Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/Shape:output:0Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Qmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2E
Cmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1 
>model/transformer_block_1/sequential_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2@
>model/transformer_block_1/sequential_1/dense_3/Tensordot/ConstЉ
=model/transformer_block_1/sequential_1/dense_3/Tensordot/ProdProdJmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2?
=model/transformer_block_1/sequential_1/dense_3/Tensordot/Prodќ
@model/transformer_block_1/sequential_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block_1/sequential_1/dense_3/Tensordot/Const_1ƒ
?model/transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1ProdLmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2_1:output:0Imodel/transformer_block_1/sequential_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2A
?model/transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1ќ
Dmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat/axisЫ
?model/transformer_block_1/sequential_1/dense_3/Tensordot/concatConcatV2Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/free:output:0Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/axes:output:0Mmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2A
?model/transformer_block_1/sequential_1/dense_3/Tensordot/concat»
>model/transformer_block_1/sequential_1/dense_3/Tensordot/stackPackFmodel/transformer_block_1/sequential_1/dense_3/Tensordot/Prod:output:0Hmodel/transformer_block_1/sequential_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2@
>model/transformer_block_1/sequential_1/dense_3/Tensordot/stackЎ
Bmodel/transformer_block_1/sequential_1/dense_3/Tensordot/transpose	TransposeAmodel/transformer_block_1/sequential_1/dense_2/Relu:activations:0Hmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2D
Bmodel/transformer_block_1/sequential_1/dense_3/Tensordot/transposeџ
@model/transformer_block_1/sequential_1/dense_3/Tensordot/ReshapeReshapeFmodel/transformer_block_1/sequential_1/dense_3/Tensordot/transpose:y:0Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2B
@model/transformer_block_1/sequential_1/dense_3/Tensordot/ReshapeЏ
?model/transformer_block_1/sequential_1/dense_3/Tensordot/MatMulMatMulImodel/transformer_block_1/sequential_1/dense_3/Tensordot/Reshape:output:0Omodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2A
?model/transformer_block_1/sequential_1/dense_3/Tensordot/MatMulќ
@model/transformer_block_1/sequential_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2B
@model/transformer_block_1/sequential_1/dense_3/Tensordot/Const_2“
Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axis®
Amodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1ConcatV2Jmodel/transformer_block_1/sequential_1/dense_3/Tensordot/GatherV2:output:0Imodel/transformer_block_1/sequential_1/dense_3/Tensordot/Const_2:output:0Omodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2C
Amodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1ћ
8model/transformer_block_1/sequential_1/dense_3/TensordotReshapeImodel/transformer_block_1/sequential_1/dense_3/Tensordot/MatMul:product:0Jmodel/transformer_block_1/sequential_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2:
8model/transformer_block_1/sequential_1/dense_3/TensordotЩ
Emodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOpNmodel_transformer_block_1_sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02G
Emodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp√
6model/transformer_block_1/sequential_1/dense_3/BiasAddBiasAddAmodel/transformer_block_1/sequential_1/dense_3/Tensordot:output:0Mmodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 28
6model/transformer_block_1/sequential_1/dense_3/BiasAddя
,model/transformer_block_1/dropout_3/IdentityIdentity?model/transformer_block_1/sequential_1/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2.
,model/transformer_block_1/dropout_3/Identityэ
model/transformer_block_1/add_1AddV2Cmodel/transformer_block_1/layer_normalization_2/batchnorm/add_1:z:05model/transformer_block_1/dropout_3/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2!
model/transformer_block_1/add_1к
Nmodel/transformer_block_1/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
Nmodel/transformer_block_1/layer_normalization_3/moments/mean/reduction_indices…
<model/transformer_block_1/layer_normalization_3/moments/meanMean#model/transformer_block_1/add_1:z:0Wmodel/transformer_block_1/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2>
<model/transformer_block_1/layer_normalization_3/moments/meanЩ
Dmodel/transformer_block_1/layer_normalization_3/moments/StopGradientStopGradientEmodel/transformer_block_1/layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2F
Dmodel/transformer_block_1/layer_normalization_3/moments/StopGradient’
Imodel/transformer_block_1/layer_normalization_3/moments/SquaredDifferenceSquaredDifference#model/transformer_block_1/add_1:z:0Mmodel/transformer_block_1/layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2K
Imodel/transformer_block_1/layer_normalization_3/moments/SquaredDifferenceт
Rmodel/transformer_block_1/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2T
Rmodel/transformer_block_1/layer_normalization_3/moments/variance/reduction_indices€
@model/transformer_block_1/layer_normalization_3/moments/varianceMeanMmodel/transformer_block_1/layer_normalization_3/moments/SquaredDifference:z:0[model/transformer_block_1/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€-*
	keep_dims(2B
@model/transformer_block_1/layer_normalization_3/moments/variance«
?model/transformer_block_1/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52A
?model/transformer_block_1/layer_normalization_3/batchnorm/add/y“
=model/transformer_block_1/layer_normalization_3/batchnorm/addAddV2Imodel/transformer_block_1/layer_normalization_3/moments/variance:output:0Hmodel/transformer_block_1/layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€-2?
=model/transformer_block_1/layer_normalization_3/batchnorm/addД
?model/transformer_block_1/layer_normalization_3/batchnorm/RsqrtRsqrtAmodel/transformer_block_1/layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€-2A
?model/transformer_block_1/layer_normalization_3/batchnorm/RsqrtЃ
Lmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpUmodel_transformer_block_1_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02N
Lmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp÷
=model/transformer_block_1/layer_normalization_3/batchnorm/mulMulCmodel/transformer_block_1/layer_normalization_3/batchnorm/Rsqrt:y:0Tmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2?
=model/transformer_block_1/layer_normalization_3/batchnorm/mulІ
?model/transformer_block_1/layer_normalization_3/batchnorm/mul_1Mul#model/transformer_block_1/add_1:z:0Amodel/transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2A
?model/transformer_block_1/layer_normalization_3/batchnorm/mul_1…
?model/transformer_block_1/layer_normalization_3/batchnorm/mul_2MulEmodel/transformer_block_1/layer_normalization_3/moments/mean:output:0Amodel/transformer_block_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2A
?model/transformer_block_1/layer_normalization_3/batchnorm/mul_2Ґ
Hmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOpQmodel_transformer_block_1_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp“
=model/transformer_block_1/layer_normalization_3/batchnorm/subSubPmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp:value:0Cmodel/transformer_block_1/layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2?
=model/transformer_block_1/layer_normalization_3/batchnorm/sub…
?model/transformer_block_1/layer_normalization_3/batchnorm/add_1AddV2Cmodel/transformer_block_1/layer_normalization_3/batchnorm/mul_1:z:0Amodel/transformer_block_1/layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€- 2A
?model/transformer_block_1/layer_normalization_3/batchnorm/add_1∞
5model/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :27
5model/global_average_pooling1d/Mean/reduction_indicesЙ
#model/global_average_pooling1d/MeanMeanCmodel/transformer_block_1/layer_normalization_3/batchnorm/add_1:z:0>model/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#model/global_average_pooling1d/MeanА
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axisЏ
model/concatenate/concatConcatV2,model/global_average_pooling1d/Mean:output:0input_2&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€(2
model/concatenate/concatЈ
#model/dense_4/MatMul/ReadVariableOpReadVariableOp,model_dense_4_matmul_readvariableop_resource*
_output_shapes

:( *
dtype02%
#model/dense_4/MatMul/ReadVariableOpЄ
model/dense_4/MatMulMatMul!model/concatenate/concat:output:0+model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
model/dense_4/MatMulґ
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/dense_4/BiasAdd/ReadVariableOpє
model/dense_4/BiasAddBiasAddmodel/dense_4/MatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
model/dense_4/BiasAddВ
model/dense_4/ReluRelumodel/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
model/dense_4/ReluФ
model/dropout_4/IdentityIdentity model/dense_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
model/dropout_4/IdentityЈ
#model/dense_5/MatMul/ReadVariableOpReadVariableOp,model_dense_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#model/dense_5/MatMul/ReadVariableOpЄ
model/dense_5/MatMulMatMul!model/dropout_4/Identity:output:0+model/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
model/dense_5/MatMulґ
$model/dense_5/BiasAdd/ReadVariableOpReadVariableOp-model_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/dense_5/BiasAdd/ReadVariableOpє
model/dense_5/BiasAddBiasAddmodel/dense_5/MatMul:product:0,model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
model/dense_5/BiasAddВ
model/dense_5/ReluRelumodel/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
model/dense_5/ReluФ
model/dropout_5/IdentityIdentity model/dense_5/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
model/dropout_5/IdentityЈ
#model/dense_6/MatMul/ReadVariableOpReadVariableOp,model_dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#model/dense_6/MatMul/ReadVariableOpЄ
model/dense_6/MatMulMatMul!model/dropout_5/Identity:output:0+model/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model/dense_6/MatMulґ
$model/dense_6/BiasAdd/ReadVariableOpReadVariableOp-model_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_6/BiasAdd/ReadVariableOpє
model/dense_6/BiasAddBiasAddmodel/dense_6/MatMul:product:0,model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model/dense_6/BiasAddЌ
IdentityIdentitymodel/dense_6/BiasAdd:output:0%^model/dense_4/BiasAdd/ReadVariableOp$^model/dense_4/MatMul/ReadVariableOp%^model/dense_5/BiasAdd/ReadVariableOp$^model/dense_5/MatMul/ReadVariableOp%^model/dense_6/BiasAdd/ReadVariableOp$^model/dense_6/MatMul/ReadVariableOp>^model/token_and_position_embedding/embedding/embedding_lookup@^model/token_and_position_embedding/embedding_1/embedding_lookupI^model/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpM^model/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpI^model/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpM^model/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpU^model/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp_^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpH^model/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOpR^model/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpJ^model/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpT^model/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpJ^model/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpT^model/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpF^model/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpH^model/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpF^model/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpH^model/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ь
_input_shapesК
З:€€€€€€€€€Љi:€€€€€€€€€::::::::::::::::::::::::2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2J
#model/dense_4/MatMul/ReadVariableOp#model/dense_4/MatMul/ReadVariableOp2L
$model/dense_5/BiasAdd/ReadVariableOp$model/dense_5/BiasAdd/ReadVariableOp2J
#model/dense_5/MatMul/ReadVariableOp#model/dense_5/MatMul/ReadVariableOp2L
$model/dense_6/BiasAdd/ReadVariableOp$model/dense_6/BiasAdd/ReadVariableOp2J
#model/dense_6/MatMul/ReadVariableOp#model/dense_6/MatMul/ReadVariableOp2~
=model/token_and_position_embedding/embedding/embedding_lookup=model/token_and_position_embedding/embedding/embedding_lookup2В
?model/token_and_position_embedding/embedding_1/embedding_lookup?model/token_and_position_embedding/embedding_1/embedding_lookup2Ф
Hmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOpHmodel/transformer_block_1/layer_normalization_2/batchnorm/ReadVariableOp2Ь
Lmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOpLmodel/transformer_block_1/layer_normalization_2/batchnorm/mul/ReadVariableOp2Ф
Hmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOpHmodel/transformer_block_1/layer_normalization_3/batchnorm/ReadVariableOp2Ь
Lmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOpLmodel/transformer_block_1/layer_normalization_3/batchnorm/mul/ReadVariableOp2ђ
Tmodel/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOpTmodel/transformer_block_1/multi_head_attention_1/attention_output/add/ReadVariableOp2ј
^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp^model/transformer_block_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2Т
Gmodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOpGmodel/transformer_block_1/multi_head_attention_1/key/add/ReadVariableOp2¶
Qmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpQmodel/transformer_block_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2Ц
Imodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOpImodel/transformer_block_1/multi_head_attention_1/query/add/ReadVariableOp2™
Smodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpSmodel/transformer_block_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2Ц
Imodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOpImodel/transformer_block_1/multi_head_attention_1/value/add/ReadVariableOp2™
Smodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpSmodel/transformer_block_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2О
Emodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOpEmodel/transformer_block_1/sequential_1/dense_2/BiasAdd/ReadVariableOp2Т
Gmodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOpGmodel/transformer_block_1/sequential_1/dense_2/Tensordot/ReadVariableOp2О
Emodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOpEmodel/transformer_block_1/sequential_1/dense_3/BiasAdd/ReadVariableOp2Т
Gmodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOpGmodel/transformer_block_1/sequential_1/dense_3/Tensordot/ReadVariableOp:Q M
(
_output_shapes
:€€€€€€€€€Љi
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_2
Ю
W
+__inference_concatenate_layer_call_fn_90363
inputs_0
inputs_1
identity‘
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_889372
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:€€€€€€€€€ :€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
ЎH
¶
G__inference_sequential_1_layer_call_and_return_conditional_losses_90590

inputs-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource-
)dense_3_tensordot_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityИҐdense_2/BiasAdd/ReadVariableOpҐ dense_2/Tensordot/ReadVariableOpҐdense_3/BiasAdd/ReadVariableOpҐ dense_3/Tensordot/ReadVariableOpЃ
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
dense_2/Tensordot/axesБ
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
dense_2/Tensordot/ShapeД
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axisщ
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2И
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis€
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
dense_2/Tensordot/Const†
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/ProdА
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1®
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1А
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axisЎ
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concatђ
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack®
dense_2/Tensordot/transpose	Transposeinputs!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dense_2/Tensordot/transposeњ
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_2/Tensordot/ReshapeЊ
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_2/Tensordot/MatMulА
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_2Д
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axisе
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1∞
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dense_2/Tensordot§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_2/BiasAdd/ReadVariableOpІ
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dense_2/BiasAddt
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dense_2/ReluЃ
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
dense_3/Tensordot/axesБ
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
dense_3/Tensordot/ShapeД
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/GatherV2/axisщ
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2И
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axis€
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
dense_3/Tensordot/Const†
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/ProdА
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1®
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1А
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axisЎ
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concatђ
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stackЉ
dense_3/Tensordot/transpose	Transposedense_2/Relu:activations:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dense_3/Tensordot/transposeњ
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_3/Tensordot/ReshapeЊ
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_3/Tensordot/MatMulА
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_2Д
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axisе
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1∞
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dense_3/Tensordot§
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOpІ
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
dense_3/BiasAddш
IdentityIdentitydense_3/BiasAdd:output:0^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€- ::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€- 
 
_user_specified_nameinputs
л
|
'__inference_dense_2_layer_call_fn_90656

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_883372
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€- ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€- 
 
_user_specified_nameinputs
М
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_90395

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeј
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seed*2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
«
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_89047

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ґ
Я
,__inference_sequential_1_layer_call_fn_90603

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_884312
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€- ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€- 
 
_user_specified_nameinputs
М
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_90442

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeј
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype0*

seed*2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
м	
џ
B__inference_dense_5_layer_call_and_return_conditional_losses_89014

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ћ
¶
,__inference_sequential_1_layer_call_fn_88469
dense_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCalldense_2_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_884582
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€- ::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:€€€€€€€€€- 
'
_user_specified_namedense_2_input
м	
џ
B__inference_dense_4_layer_call_and_return_conditional_losses_90374

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:( *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€(
 
_user_specified_nameinputs
ѓ 
б
B__inference_dense_2_layer_call_and_return_conditional_losses_88337

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackР
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Р
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЗ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
ReluЮ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€- ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€- 
 
_user_specified_nameinputs
Љ
б
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
identityИҐStatefulPartitionedCallЂ
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
:€€€€€€€€€*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_893342
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ь
_input_shapesК
З:€€€€€€€€€Љi:€€€€€€€€€::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:€€€€€€€€€Љi
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
м	
џ
B__inference_dense_5_layer_call_and_return_conditional_losses_90421

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
џ
|
'__inference_dense_4_layer_call_fn_90383

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_889572
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€(::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€(
 
_user_specified_nameinputs
ю
M
1__inference_average_pooling1d_layer_call_fn_88302

inputs
identityг
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_882962
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ї
T
8__inference_global_average_pooling1d_layer_call_fn_90350

inputs
identity‘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_889232
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€- :S O
+
_output_shapes
:€€€€€€€€€- 
 
_user_specified_nameinputs
с
T
8__inference_global_average_pooling1d_layer_call_fn_90339

inputs
identityЁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_884852
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ћ
¶
,__inference_sequential_1_layer_call_fn_88442
dense_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCalldense_2_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_884312
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€- ::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:€€€€€€€€€- 
'
_user_specified_namedense_2_input
«
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_88990

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
џ
|
'__inference_dense_5_layer_call_fn_90430

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_890142
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
м	
џ
B__inference_dense_4_layer_call_and_return_conditional_losses_88957

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:( *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€(
 
_user_specified_nameinputs
ќ

я
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
identityИҐStatefulPartitionedCallј
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
:€€€€€€€€€- *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_886822
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€- ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€- 
 
_user_specified_nameinputs
ў
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
:€€€€€€€€€ 2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€- :S O
+
_output_shapes
:€€€€€€€€€- 
 
_user_specified_nameinputs
¬5
к
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
identityИҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐ4token_and_position_embedding/StatefulPartitionedCallҐ+transformer_block_1/StatefulPartitionedCallю
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1"token_and_position_embedding_89091"token_and_position_embedding_89093*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Љi *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_8851726
4token_and_position_embedding/StatefulPartitionedCallђ
!average_pooling1d/PartitionedCallPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_882962#
!average_pooling1d/PartitionedCallЙ
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0transformer_block_1_89097transformer_block_1_89099transformer_block_1_89101transformer_block_1_89103transformer_block_1_89105transformer_block_1_89107transformer_block_1_89109transformer_block_1_89111transformer_block_1_89113transformer_block_1_89115transformer_block_1_89117transformer_block_1_89119transformer_block_1_89121transformer_block_1_89123transformer_block_1_89125transformer_block_1_89127*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_888092-
+transformer_block_1/StatefulPartitionedCallі
(global_average_pooling1d/PartitionedCallPartitionedCall4transformer_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_889232*
(global_average_pooling1d/PartitionedCallФ
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_889372
concatenate/PartitionedCall≠
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_89132dense_4_89134*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_889572!
dense_4/StatefulPartitionedCallы
dropout_4/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_889902
dropout_4/PartitionedCallЂ
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_5_89138dense_5_89140*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_890142!
dense_5/StatefulPartitionedCallы
dropout_5/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_890472
dropout_5/PartitionedCallЂ
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_6_89144dense_6_89146*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_890702!
dense_6/StatefulPartitionedCall«
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ь
_input_shapesК
З:€€€€€€€€€Љi:€€€€€€€€€::::::::::::::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€Љi
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_2
Р	
џ
B__inference_dense_6_layer_call_and_return_conditional_losses_90467

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
«у
ш$
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
identity_55ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ѕ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*Ќ
value√Bј7B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/7/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/8/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesэ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*Б
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЅ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*т
_output_shapesя
№:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЮ
AssignVariableOpAssignVariableOpassignvariableop_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¶
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¶
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5§
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_6_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Э
AssignVariableOp_6AssignVariableOpassignvariableop_6_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7•
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8†
AssignVariableOp_8AssignVariableOpassignvariableop_8_momentumIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_9†
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_iterIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ќ
AssignVariableOp_10AssignVariableOpEassignvariableop_10_token_and_position_embedding_embedding_embeddingsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ѕ
AssignVariableOp_11AssignVariableOpGassignvariableop_11_token_and_position_embedding_embedding_1_embeddingsIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12”
AssignVariableOp_12AssignVariableOpKassignvariableop_12_transformer_block_1_multi_head_attention_1_query_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13—
AssignVariableOp_13AssignVariableOpIassignvariableop_13_transformer_block_1_multi_head_attention_1_query_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14—
AssignVariableOp_14AssignVariableOpIassignvariableop_14_transformer_block_1_multi_head_attention_1_key_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ѕ
AssignVariableOp_15AssignVariableOpGassignvariableop_15_transformer_block_1_multi_head_attention_1_key_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16”
AssignVariableOp_16AssignVariableOpKassignvariableop_16_transformer_block_1_multi_head_attention_1_value_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17—
AssignVariableOp_17AssignVariableOpIassignvariableop_17_transformer_block_1_multi_head_attention_1_value_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ё
AssignVariableOp_18AssignVariableOpVassignvariableop_18_transformer_block_1_multi_head_attention_1_attention_output_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19№
AssignVariableOp_19AssignVariableOpTassignvariableop_19_transformer_block_1_multi_head_attention_1_attention_output_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20™
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_2_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21®
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_2_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22™
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_3_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23®
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_3_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ћ
AssignVariableOp_24AssignVariableOpCassignvariableop_24_transformer_block_1_layer_normalization_2_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25 
AssignVariableOp_25AssignVariableOpBassignvariableop_25_transformer_block_1_layer_normalization_2_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ћ
AssignVariableOp_26AssignVariableOpCassignvariableop_26_transformer_block_1_layer_normalization_3_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27 
AssignVariableOp_27AssignVariableOpBassignvariableop_27_transformer_block_1_layer_normalization_3_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28°
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29°
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ј
AssignVariableOp_30AssignVariableOp/assignvariableop_30_sgd_dense_4_kernel_momentumIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31µ
AssignVariableOp_31AssignVariableOp-assignvariableop_31_sgd_dense_4_bias_momentumIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ј
AssignVariableOp_32AssignVariableOp/assignvariableop_32_sgd_dense_5_kernel_momentumIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33µ
AssignVariableOp_33AssignVariableOp-assignvariableop_33_sgd_dense_5_bias_momentumIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ј
AssignVariableOp_34AssignVariableOp/assignvariableop_34_sgd_dense_6_kernel_momentumIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35µ
AssignVariableOp_35AssignVariableOp-assignvariableop_35_sgd_dense_6_bias_momentumIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Џ
AssignVariableOp_36AssignVariableOpRassignvariableop_36_sgd_token_and_position_embedding_embedding_embeddings_momentumIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37№
AssignVariableOp_37AssignVariableOpTassignvariableop_37_sgd_token_and_position_embedding_embedding_1_embeddings_momentumIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38а
AssignVariableOp_38AssignVariableOpXassignvariableop_38_sgd_transformer_block_1_multi_head_attention_1_query_kernel_momentumIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39ё
AssignVariableOp_39AssignVariableOpVassignvariableop_39_sgd_transformer_block_1_multi_head_attention_1_query_bias_momentumIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40ё
AssignVariableOp_40AssignVariableOpVassignvariableop_40_sgd_transformer_block_1_multi_head_attention_1_key_kernel_momentumIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41№
AssignVariableOp_41AssignVariableOpTassignvariableop_41_sgd_transformer_block_1_multi_head_attention_1_key_bias_momentumIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42а
AssignVariableOp_42AssignVariableOpXassignvariableop_42_sgd_transformer_block_1_multi_head_attention_1_value_kernel_momentumIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43ё
AssignVariableOp_43AssignVariableOpVassignvariableop_43_sgd_transformer_block_1_multi_head_attention_1_value_bias_momentumIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44л
AssignVariableOp_44AssignVariableOpcassignvariableop_44_sgd_transformer_block_1_multi_head_attention_1_attention_output_kernel_momentumIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45й
AssignVariableOp_45AssignVariableOpaassignvariableop_45_sgd_transformer_block_1_multi_head_attention_1_attention_output_bias_momentumIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Ј
AssignVariableOp_46AssignVariableOp/assignvariableop_46_sgd_dense_2_kernel_momentumIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47µ
AssignVariableOp_47AssignVariableOp-assignvariableop_47_sgd_dense_2_bias_momentumIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Ј
AssignVariableOp_48AssignVariableOp/assignvariableop_48_sgd_dense_3_kernel_momentumIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49µ
AssignVariableOp_49AssignVariableOp-assignvariableop_49_sgd_dense_3_bias_momentumIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Ў
AssignVariableOp_50AssignVariableOpPassignvariableop_50_sgd_transformer_block_1_layer_normalization_2_gamma_momentumIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51„
AssignVariableOp_51AssignVariableOpOassignvariableop_51_sgd_transformer_block_1_layer_normalization_2_beta_momentumIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Ў
AssignVariableOp_52AssignVariableOpPassignvariableop_52_sgd_transformer_block_1_layer_normalization_3_gamma_momentumIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53„
AssignVariableOp_53AssignVariableOpOassignvariableop_53_sgd_transformer_block_1_layer_normalization_3_beta_momentumIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_539
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpВ

Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_54х	
Identity_55IdentityIdentity_54:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_55"#
identity_55Identity_55:output:0*п
_input_shapesЁ
Џ: ::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
«
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_90447

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Х
E
)__inference_dropout_4_layer_call_fn_90410

inputs
identity≈
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_889902
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
∆8
≤
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
identityИҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐ!dropout_4/StatefulPartitionedCallҐ!dropout_5/StatefulPartitionedCallҐ4token_and_position_embedding/StatefulPartitionedCallҐ+transformer_block_1/StatefulPartitionedCallю
4token_and_position_embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1"token_and_position_embedding_88528"token_and_position_embedding_88530*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Љi *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *`
f[RY
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_8851726
4token_and_position_embedding/StatefulPartitionedCallђ
!average_pooling1d/PartitionedCallPartitionedCall=token_and_position_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_882962#
!average_pooling1d/PartitionedCallЙ
+transformer_block_1/StatefulPartitionedCallStatefulPartitionedCall*average_pooling1d/PartitionedCall:output:0transformer_block_1_88885transformer_block_1_88887transformer_block_1_88889transformer_block_1_88891transformer_block_1_88893transformer_block_1_88895transformer_block_1_88897transformer_block_1_88899transformer_block_1_88901transformer_block_1_88903transformer_block_1_88905transformer_block_1_88907transformer_block_1_88909transformer_block_1_88911transformer_block_1_88913transformer_block_1_88915*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€- *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_886822-
+transformer_block_1/StatefulPartitionedCallі
(global_average_pooling1d/PartitionedCallPartitionedCall4transformer_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_889232*
(global_average_pooling1d/PartitionedCallФ
concatenate/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_889372
concatenate/PartitionedCall≠
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_88968dense_4_88970*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_889572!
dense_4/StatefulPartitionedCallУ
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_889852#
!dropout_4/StatefulPartitionedCall≥
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_5_89025dense_5_89027*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_890142!
dense_5/StatefulPartitionedCallЈ
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_890422#
!dropout_5/StatefulPartitionedCall≥
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_6_89081dense_6_89083*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_890702!
dense_6/StatefulPartitionedCallП
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall5^token_and_position_embedding/StatefulPartitionedCall,^transformer_block_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ь
_input_shapesК
З:€€€€€€€€€Љi:€€€€€€€€€::::::::::::::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2l
4token_and_position_embedding/StatefulPartitionedCall4token_and_position_embedding/StatefulPartitionedCall2Z
+transformer_block_1/StatefulPartitionedCall+transformer_block_1/StatefulPartitionedCall:Q M
(
_output_shapes
:€€€€€€€€€Љi
!
_user_specified_name	input_1:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_2
ќ

я
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
identityИҐStatefulPartitionedCallј
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
:€€€€€€€€€- *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_888092
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€- ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€- 
 
_user_specified_nameinputs
ѕ
б
B__inference_dense_3_layer_call_and_return_conditional_losses_88383

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackР
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Р
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЗ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2	
BiasAddЬ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€- ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€- 
 
_user_specified_nameinputs
ѓ 
б
B__inference_dense_2_layer_call_and_return_conditional_losses_90647

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackР
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Р
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЗ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€- 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€- 2
ReluЮ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€- 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€- ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€- 
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*и
serving_default‘
<
input_11
serving_default_input_1:0€€€€€€€€€Љi
;
input_20
serving_default_input_2:0€€€€€€€€€;
dense_60
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:§≤
∞)
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
’_default_save_signature
÷__call__
+„&call_and_return_all_conditional_losses"¬%
_tf_keras_network¶%{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 13500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d", "inbound_nodes": [[["token_and_position_embedding", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_1", "inbound_nodes": [[["average_pooling1d", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling1d", "inbound_nodes": [[["transformer_block_1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["global_average_pooling1d", 0, 0, {}], ["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dropout_4", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_5", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dropout_5", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["dense_6", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 13500]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 13500]}, {"class_name": "TensorShape", "items": [null, 8]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.004999999888241291, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
с"о
_tf_keras_input_layerќ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 13500]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 13500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
е
	token_emb
pos_emb
trainable_variables
	variables
regularization_losses
	keras_api
Ў__call__
+ў&call_and_return_all_conditional_losses"Є
_tf_keras_layerЮ{"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
З
trainable_variables
	variables
regularization_losses
	keras_api
Џ__call__
+џ&call_and_return_all_conditional_losses"ц
_tf_keras_layer№{"class_name": "AveragePooling1D", "name": "average_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Д
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
№__call__
+Ё&call_and_return_all_conditional_losses"•
_tf_keras_layerЛ{"class_name": "TransformerBlock", "name": "transformer_block_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Х
'trainable_variables
(	variables
)regularization_losses
*	keras_api
ё__call__
+я&call_and_return_all_conditional_losses"Д
_tf_keras_layerк{"class_name": "GlobalAveragePooling1D", "name": "global_average_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
й"ж
_tf_keras_input_layer∆{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
 
+trainable_variables
,	variables
-regularization_losses
.	keras_api
а__call__
+б&call_and_return_all_conditional_losses"є
_tf_keras_layerЯ{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32]}, {"class_name": "TensorShape", "items": [null, 8]}]}
т

/kernel
0bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
в__call__
+г&call_and_return_all_conditional_losses"Ћ
_tf_keras_layer±{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40]}}
з
5trainable_variables
6	variables
7regularization_losses
8	keras_api
д__call__
+е&call_and_return_all_conditional_losses"÷
_tf_keras_layerЉ{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
т

9kernel
:bias
;trainable_variables
<	variables
=regularization_losses
>	keras_api
ж__call__
+з&call_and_return_all_conditional_losses"Ћ
_tf_keras_layer±{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
з
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
и__call__
+й&call_and_return_all_conditional_losses"÷
_tf_keras_layerЉ{"class_name": "Dropout", "name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
у

Ckernel
Dbias
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
к__call__
+л&call_and_return_all_conditional_losses"ћ
_tf_keras_layer≤{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
б
	Idecay
Jlearning_rate
Kmomentum
Liter/momentumљ0momentumЊ9momentumњ:momentumјCmomentumЅDmomentum¬Mmomentum√NmomentumƒOmomentum≈Pmomentum∆Qmomentum«Rmomentum»Smomentum…Tmomentum UmomentumЋVmomentumћWmomentumЌXmomentumќYmomentumѕZmomentum–[momentum—\momentum“]momentum”^momentum‘"
	optimizer
÷
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
÷
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
ќ
_non_trainable_variables
`metrics
alayer_metrics
trainable_variables
blayer_regularization_losses

clayers
	variables
regularization_losses
÷__call__
’_default_save_signature
+„&call_and_return_all_conditional_losses
'„"call_and_return_conditional_losses"
_generic_user_object
-
мserving_default"
signature_map
ђ
M
embeddings
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
н__call__
+о&call_and_return_all_conditional_losses"Л
_tf_keras_layerс{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 5, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13500]}}
≠
N
embeddings
htrainable_variables
i	variables
jregularization_losses
k	keras_api
п__call__
+р&call_and_return_all_conditional_losses"М
_tf_keras_layerт{"class_name": "Embedding", "name": "embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13500, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
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
∞
lnon_trainable_variables
mmetrics
nlayer_metrics
trainable_variables
olayer_regularization_losses

players
	variables
regularization_losses
Ў__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
qnon_trainable_variables
rmetrics
slayer_metrics
trainable_variables
tlayer_regularization_losses

ulayers
	variables
regularization_losses
Џ__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
ю
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
с__call__
+т&call_and_return_all_conditional_losses"Д
_tf_keras_layerк{"class_name": "MultiHeadAttention", "name": "multi_head_attention_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_1", "trainable": true, "dtype": "float32", "num_heads": 1, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
©
Аlayer_with_weights-0
Аlayer-0
Бlayer_with_weights-1
Бlayer-1
Вtrainable_variables
Г	variables
Дregularization_losses
Е	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"¬
_tf_keras_sequential£{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 45, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_2_input"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 45, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_2_input"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
и
	Жaxis
	[gamma
\beta
Зtrainable_variables
И	variables
Йregularization_losses
К	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"≥
_tf_keras_layerЩ{"class_name": "LayerNormalization", "name": "layer_normalization_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 32]}}
и
	Лaxis
	]gamma
^beta
Мtrainable_variables
Н	variables
Оregularization_losses
П	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses"≥
_tf_keras_layerЩ{"class_name": "LayerNormalization", "name": "layer_normalization_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_3", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 32]}}
л
Рtrainable_variables
С	variables
Тregularization_losses
У	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses"÷
_tf_keras_layerЉ{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
л
Фtrainable_variables
Х	variables
Цregularization_losses
Ч	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses"÷
_tf_keras_layerЉ{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
Ц
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
Ц
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
µ
Шnon_trainable_variables
Щmetrics
Ъlayer_metrics
#trainable_variables
 Ыlayer_regularization_losses
Ьlayers
$	variables
%regularization_losses
№__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Эnon_trainable_variables
Юmetrics
Яlayer_metrics
'trainable_variables
 †layer_regularization_losses
°layers
(	variables
)regularization_losses
ё__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ґnon_trainable_variables
£metrics
§layer_metrics
+trainable_variables
 •layer_regularization_losses
¶layers
,	variables
-regularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
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
µ
Іnon_trainable_variables
®metrics
©layer_metrics
1trainable_variables
 ™layer_regularization_losses
Ђlayers
2	variables
3regularization_losses
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ђnon_trainable_variables
≠metrics
Ѓlayer_metrics
5trainable_variables
 ѓlayer_regularization_losses
∞layers
6	variables
7regularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
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
µ
±non_trainable_variables
≤metrics
≥layer_metrics
;trainable_variables
 іlayer_regularization_losses
µlayers
<	variables
=regularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ґnon_trainable_variables
Јmetrics
Єlayer_metrics
?trainable_variables
 єlayer_regularization_losses
Їlayers
@	variables
Aregularization_losses
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
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
µ
їnon_trainable_variables
Љmetrics
љlayer_metrics
Etrainable_variables
 Њlayer_regularization_losses
њlayers
F	variables
Gregularization_losses
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
C:A 21token_and_position_embedding/embedding/embeddings
F:D	Љi 23token_and_position_embedding/embedding_1/embeddings
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
ј0"
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
µ
Ѕnon_trainable_variables
¬metrics
√layer_metrics
dtrainable_variables
 ƒlayer_regularization_losses
≈layers
e	variables
fregularization_losses
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
'
N0"
trackable_list_wrapper
'
N0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
∆non_trainable_variables
«metrics
»layer_metrics
htrainable_variables
 …layer_regularization_losses
 layers
i	variables
jregularization_losses
п__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
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
…
Ћpartial_output_shape
ћfull_output_shape

Okernel
Pbias
Ќtrainable_variables
ќ	variables
ѕregularization_losses
–	keras_api
э__call__
+ю&call_and_return_all_conditional_losses"л
_tf_keras_layer—{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 32]}}
≈
—partial_output_shape
“full_output_shape

Qkernel
Rbias
”trainable_variables
‘	variables
’regularization_losses
÷	keras_api
€__call__
+А&call_and_return_all_conditional_losses"з
_tf_keras_layerЌ{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 32]}}
…
„partial_output_shape
Ўfull_output_shape

Skernel
Tbias
ўtrainable_variables
Џ	variables
џregularization_losses
№	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"л
_tf_keras_layer—{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 32]}}
л
Ёtrainable_variables
ё	variables
яregularization_losses
а	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"÷
_tf_keras_layerЉ{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
з
бtrainable_variables
в	variables
гregularization_losses
д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"“
_tf_keras_layerЄ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
ё
еpartial_output_shape
жfull_output_shape

Ukernel
Vbias
зtrainable_variables
и	variables
йregularization_losses
к	keras_api
З__call__
+И&call_and_return_all_conditional_losses"А
_tf_keras_layerж{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 32], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 1, 32]}}
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
µ
лnon_trainable_variables
мmetrics
нlayer_metrics
|trainable_variables
 оlayer_regularization_losses
пlayers
}	variables
~regularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
ъ

Wkernel
Xbias
рtrainable_variables
с	variables
тregularization_losses
у	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"ѕ
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 32]}}
ь

Ykernel
Zbias
фtrainable_variables
х	variables
цregularization_losses
ч	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"—
_tf_keras_layerЈ{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 45, 32]}}
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
Є
шnon_trainable_variables
щmetrics
ъlayer_metrics
Вtrainable_variables
 ыlayer_regularization_losses
ьlayers
Г	variables
Дregularization_losses
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
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
Є
эnon_trainable_variables
юmetrics
€layer_metrics
Зtrainable_variables
 Аlayer_regularization_losses
Бlayers
И	variables
Йregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
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
Є
Вnon_trainable_variables
Гmetrics
Дlayer_metrics
Мtrainable_variables
 Еlayer_regularization_losses
Жlayers
Н	variables
Оregularization_losses
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Зnon_trainable_variables
Иmetrics
Йlayer_metrics
Рtrainable_variables
 Кlayer_regularization_losses
Лlayers
С	variables
Тregularization_losses
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Мnon_trainable_variables
Нmetrics
Оlayer_metrics
Фtrainable_variables
 Пlayer_regularization_losses
Рlayers
Х	variables
Цregularization_losses
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
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
њ

Сtotal

Тcount
У	variables
Ф	keras_api"Д
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
Є
Хnon_trainable_variables
Цmetrics
Чlayer_metrics
Ќtrainable_variables
 Шlayer_regularization_losses
Щlayers
ќ	variables
ѕregularization_losses
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
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
Є
Ъnon_trainable_variables
Ыmetrics
Ьlayer_metrics
”trainable_variables
 Эlayer_regularization_losses
Юlayers
‘	variables
’regularization_losses
€__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
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
Є
Яnon_trainable_variables
†metrics
°layer_metrics
ўtrainable_variables
 Ґlayer_regularization_losses
£layers
Џ	variables
џregularization_losses
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
§non_trainable_variables
•metrics
¶layer_metrics
Ёtrainable_variables
 Іlayer_regularization_losses
®layers
ё	variables
яregularization_losses
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
©non_trainable_variables
™metrics
Ђlayer_metrics
бtrainable_variables
 ђlayer_regularization_losses
≠layers
в	variables
гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
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
Є
Ѓnon_trainable_variables
ѓmetrics
∞layer_metrics
зtrainable_variables
 ±layer_regularization_losses
≤layers
и	variables
йregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
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
Є
≥non_trainable_variables
іmetrics
µlayer_metrics
рtrainable_variables
 ґlayer_regularization_losses
Јlayers
с	variables
тregularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
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
Є
Єnon_trainable_variables
єmetrics
Їlayer_metrics
фtrainable_variables
 їlayer_regularization_losses
Љlayers
х	variables
цregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
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
А0
Б1"
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
С0
Т1"
trackable_list_wrapper
.
У	variables"
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
Q:O	Љi 2@SGD/token_and_position_embedding/embedding_1/embeddings/momentum
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
З2Д
 __inference__wrapped_model_88287я
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *OҐL
JЪG
"К
input_1€€€€€€€€€Љi
!К
input_2€€€€€€€€€
в2я
%__inference_model_layer_call_fn_89946
%__inference_model_layer_call_fn_89385
%__inference_model_layer_call_fn_89892
%__inference_model_layer_call_fn_89268ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ќ2Ћ
@__inference_model_layer_call_and_return_conditional_losses_89660
@__inference_model_layer_call_and_return_conditional_losses_89838
@__inference_model_layer_call_and_return_conditional_losses_89087
@__inference_model_layer_call_and_return_conditional_losses_89150ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
б2ё
<__inference_token_and_position_embedding_layer_call_fn_89979Э
Ф≤Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ь2щ
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_89970Э
Ф≤Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
М2Й
1__inference_average_pooling1d_layer_call_fn_88302”
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *3Ґ0
.К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€
І2§
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_88296”
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *3Ґ0
.К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€
†2Э
3__inference_transformer_block_1_layer_call_fn_90328
3__inference_transformer_block_1_layer_call_fn_90291∞
І≤£
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_90254
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_90127∞
І≤£
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
©2¶
8__inference_global_average_pooling1d_layer_call_fn_90339
8__inference_global_average_pooling1d_layer_call_fn_90350ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
я2№
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_90345
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_90334ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_concatenate_layer_call_fn_90363Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_concatenate_layer_call_and_return_conditional_losses_90357Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_dense_4_layer_call_fn_90383Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_dense_4_layer_call_and_return_conditional_losses_90374Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Р2Н
)__inference_dropout_4_layer_call_fn_90405
)__inference_dropout_4_layer_call_fn_90410і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
∆2√
D__inference_dropout_4_layer_call_and_return_conditional_losses_90395
D__inference_dropout_4_layer_call_and_return_conditional_losses_90400і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
—2ќ
'__inference_dense_5_layer_call_fn_90430Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_dense_5_layer_call_and_return_conditional_losses_90421Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Р2Н
)__inference_dropout_5_layer_call_fn_90452
)__inference_dropout_5_layer_call_fn_90457і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
∆2√
D__inference_dropout_5_layer_call_and_return_conditional_losses_90447
D__inference_dropout_5_layer_call_and_return_conditional_losses_90442і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
—2ќ
'__inference_dense_6_layer_call_fn_90476Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_dense_6_layer_call_and_return_conditional_losses_90467Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—Bќ
#__inference_signature_wrapper_89447input_1input_2"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
В2€ь
у≤п
FullArgSpece
args]ЪZ
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
defaultsЪ

 

 
p 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
В2€ь
у≤п
FullArgSpece
args]ЪZ
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
defaultsЪ

 

 
p 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ю2ы
,__inference_sequential_1_layer_call_fn_88442
,__inference_sequential_1_layer_call_fn_90616
,__inference_sequential_1_layer_call_fn_90603
,__inference_sequential_1_layer_call_fn_88469ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
к2з
G__inference_sequential_1_layer_call_and_return_conditional_losses_90533
G__inference_sequential_1_layer_call_and_return_conditional_losses_88400
G__inference_sequential_1_layer_call_and_return_conditional_losses_90590
G__inference_sequential_1_layer_call_and_return_conditional_losses_88414ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
µ2≤ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
µ2≤ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_dense_2_layer_call_fn_90656Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_dense_2_layer_call_and_return_conditional_losses_90647Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_dense_3_layer_call_fn_90695Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_dense_3_layer_call_and_return_conditional_losses_90686Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 Ќ
 __inference__wrapped_model_88287®NMOPQRSTUV[\WXYZ]^/09:CDYҐV
OҐL
JЪG
"К
input_1€€€€€€€€€Љi
!К
input_2€€€€€€€€€
™ "1™.
,
dense_6!К
dense_6€€€€€€€€€’
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_88296ДEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";Ґ8
1К.
0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ђ
1__inference_average_pooling1d_layer_call_fn_88302wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€ќ
F__inference_concatenate_layer_call_and_return_conditional_losses_90357ГZҐW
PҐM
KЪH
"К
inputs/0€€€€€€€€€ 
"К
inputs/1€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€(
Ъ •
+__inference_concatenate_layer_call_fn_90363vZҐW
PҐM
KЪH
"К
inputs/0€€€€€€€€€ 
"К
inputs/1€€€€€€€€€
™ "К€€€€€€€€€(™
B__inference_dense_2_layer_call_and_return_conditional_losses_90647dWX3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€- 
™ ")Ґ&
К
0€€€€€€€€€- 
Ъ В
'__inference_dense_2_layer_call_fn_90656WWX3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€- 
™ "К€€€€€€€€€- ™
B__inference_dense_3_layer_call_and_return_conditional_losses_90686dYZ3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€- 
™ ")Ґ&
К
0€€€€€€€€€- 
Ъ В
'__inference_dense_3_layer_call_fn_90695WYZ3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€- 
™ "К€€€€€€€€€- Ґ
B__inference_dense_4_layer_call_and_return_conditional_losses_90374\/0/Ґ,
%Ґ"
 К
inputs€€€€€€€€€(
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ z
'__inference_dense_4_layer_call_fn_90383O/0/Ґ,
%Ґ"
 К
inputs€€€€€€€€€(
™ "К€€€€€€€€€ Ґ
B__inference_dense_5_layer_call_and_return_conditional_losses_90421\9:/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ z
'__inference_dense_5_layer_call_fn_90430O9:/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€ Ґ
B__inference_dense_6_layer_call_and_return_conditional_losses_90467\CD/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€
Ъ z
'__inference_dense_6_layer_call_fn_90476OCD/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€§
D__inference_dropout_4_layer_call_and_return_conditional_losses_90395\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ §
D__inference_dropout_4_layer_call_and_return_conditional_losses_90400\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ |
)__inference_dropout_4_layer_call_fn_90405O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p
™ "К€€€€€€€€€ |
)__inference_dropout_4_layer_call_fn_90410O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p 
™ "К€€€€€€€€€ §
D__inference_dropout_5_layer_call_and_return_conditional_losses_90442\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ §
D__inference_dropout_5_layer_call_and_return_conditional_losses_90447\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ |
)__inference_dropout_5_layer_call_fn_90452O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p
™ "К€€€€€€€€€ |
)__inference_dropout_5_layer_call_fn_90457O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p 
™ "К€€€€€€€€€ “
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_90334{IҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ ".Ґ+
$К!
0€€€€€€€€€€€€€€€€€€
Ъ Ј
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_90345`7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€- 

 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ ™
8__inference_global_average_pooling1d_layer_call_fn_90339nIҐF
?Ґ<
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€

 
™ "!К€€€€€€€€€€€€€€€€€€П
8__inference_global_average_pooling1d_layer_call_fn_90350S7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€- 

 
™ "К€€€€€€€€€ й
@__inference_model_layer_call_and_return_conditional_losses_89087§NMOPQRSTUV[\WXYZ]^/09:CDaҐ^
WҐT
JЪG
"К
input_1€€€€€€€€€Љi
!К
input_2€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ й
@__inference_model_layer_call_and_return_conditional_losses_89150§NMOPQRSTUV[\WXYZ]^/09:CDaҐ^
WҐT
JЪG
"К
input_1€€€€€€€€€Љi
!К
input_2€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ л
@__inference_model_layer_call_and_return_conditional_losses_89660¶NMOPQRSTUV[\WXYZ]^/09:CDcҐ`
YҐV
LЪI
#К 
inputs/0€€€€€€€€€Љi
"К
inputs/1€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ л
@__inference_model_layer_call_and_return_conditional_losses_89838¶NMOPQRSTUV[\WXYZ]^/09:CDcҐ`
YҐV
LЪI
#К 
inputs/0€€€€€€€€€Љi
"К
inputs/1€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ѕ
%__inference_model_layer_call_fn_89268ЧNMOPQRSTUV[\WXYZ]^/09:CDaҐ^
WҐT
JЪG
"К
input_1€€€€€€€€€Љi
!К
input_2€€€€€€€€€
p

 
™ "К€€€€€€€€€Ѕ
%__inference_model_layer_call_fn_89385ЧNMOPQRSTUV[\WXYZ]^/09:CDaҐ^
WҐT
JЪG
"К
input_1€€€€€€€€€Љi
!К
input_2€€€€€€€€€
p 

 
™ "К€€€€€€€€€√
%__inference_model_layer_call_fn_89892ЩNMOPQRSTUV[\WXYZ]^/09:CDcҐ`
YҐV
LЪI
#К 
inputs/0€€€€€€€€€Љi
"К
inputs/1€€€€€€€€€
p

 
™ "К€€€€€€€€€√
%__inference_model_layer_call_fn_89946ЩNMOPQRSTUV[\WXYZ]^/09:CDcҐ`
YҐV
LЪI
#К 
inputs/0€€€€€€€€€Љi
"К
inputs/1€€€€€€€€€
p 

 
™ "К€€€€€€€€€ј
G__inference_sequential_1_layer_call_and_return_conditional_losses_88400uWXYZBҐ?
8Ґ5
+К(
dense_2_input€€€€€€€€€- 
p

 
™ ")Ґ&
К
0€€€€€€€€€- 
Ъ ј
G__inference_sequential_1_layer_call_and_return_conditional_losses_88414uWXYZBҐ?
8Ґ5
+К(
dense_2_input€€€€€€€€€- 
p 

 
™ ")Ґ&
К
0€€€€€€€€€- 
Ъ є
G__inference_sequential_1_layer_call_and_return_conditional_losses_90533nWXYZ;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€- 
p

 
™ ")Ґ&
К
0€€€€€€€€€- 
Ъ є
G__inference_sequential_1_layer_call_and_return_conditional_losses_90590nWXYZ;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€- 
p 

 
™ ")Ґ&
К
0€€€€€€€€€- 
Ъ Ш
,__inference_sequential_1_layer_call_fn_88442hWXYZBҐ?
8Ґ5
+К(
dense_2_input€€€€€€€€€- 
p

 
™ "К€€€€€€€€€- Ш
,__inference_sequential_1_layer_call_fn_88469hWXYZBҐ?
8Ґ5
+К(
dense_2_input€€€€€€€€€- 
p 

 
™ "К€€€€€€€€€- С
,__inference_sequential_1_layer_call_fn_90603aWXYZ;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€- 
p

 
™ "К€€€€€€€€€- С
,__inference_sequential_1_layer_call_fn_90616aWXYZ;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€- 
p 

 
™ "К€€€€€€€€€- б
#__inference_signature_wrapper_89447єNMOPQRSTUV[\WXYZ]^/09:CDjҐg
Ґ 
`™]
-
input_1"К
input_1€€€€€€€€€Љi
,
input_2!К
input_2€€€€€€€€€"1™.
,
dense_6!К
dense_6€€€€€€€€€Є
W__inference_token_and_position_embedding_layer_call_and_return_conditional_losses_89970]NM+Ґ(
!Ґ
К
x€€€€€€€€€Љi
™ "*Ґ'
 К
0€€€€€€€€€Љi 
Ъ Р
<__inference_token_and_position_embedding_layer_call_fn_89979PNM+Ґ(
!Ґ
К
x€€€€€€€€€Љi
™ "К€€€€€€€€€Љi »
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_90127vOPQRSTUV[\WXYZ]^7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€- 
p
™ ")Ґ&
К
0€€€€€€€€€- 
Ъ »
N__inference_transformer_block_1_layer_call_and_return_conditional_losses_90254vOPQRSTUV[\WXYZ]^7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€- 
p 
™ ")Ґ&
К
0€€€€€€€€€- 
Ъ †
3__inference_transformer_block_1_layer_call_fn_90291iOPQRSTUV[\WXYZ]^7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€- 
p
™ "К€€€€€€€€€- †
3__inference_transformer_block_1_layer_call_fn_90328iOPQRSTUV[\WXYZ]^7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€- 
p 
™ "К€€€€€€€€€- 