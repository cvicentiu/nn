require 'nn'
require 'clnn'

local model = nn.Sequential()

local function ConvBNRelu(nInputPlane, nOutputPlane)
	model:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 5, 5, 1, 1, 2, 2))
	model:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
	model:add(nn.ReLU(true))
	return model
end


ConvBNRelu(3, 64):add(nn.Dropout(0.3))
model:add(nn.SpatialMaxPooling(2, 2, 2, 2):ceil())
ConvBNRelu(64, 8):add(nn.Dropout(0.3))
model:add(nn.SpatialMaxPooling(2, 2, 2, 2):ceil())
model:add(nn.View(512))

classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512,512))
classifier:add(nn.BatchNormalization(512))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512,10))
model:add(classifier)


classifier = nn.Sequential()
classifier:add(nn.Linear(512, 10))


local function InitWeights(model)
	local function init(name)
		for k,v in pairs(model:findModules(name)) do
			local n = v.kW*v.kH*v.nOutputPlane
			v.weight:normal(0, math.sqrt(2 / n))
			v.bias:zero()
			print('DIDI')
		end
	end
	init 'nn.SpatialConvolution'
end

InitWeights(model)


return model
