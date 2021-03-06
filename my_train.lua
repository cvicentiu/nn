require 'xlua'
require 'optim'
require 'clnn'
lapp = require 'pl.lapp'
dofile './provider.lua'
local c = require 'trepl.colorize'
print(cltorch.getDeviceCount())
cltorch.setDevice(2)


opt = lapp[[
   -s,--save                  (default "my_logs")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vic)     model name
   --max_epoch                (default 300)           maximum number of iterations
]]

print(opt)

print(c.blue '==>' .. ' configuring model')

local model = nn.Sequential()
model:add(nn.Copy('torch.FloatTensor', 'torch.ClTensor'):cl())
model:add(dofile('models/'..opt.model..'.lua'):cl())
model:get(2).updateGradInput = function(input) return end
print(model)

print(c.blue '==>' .. ' loading data')
provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters, gradParameters = model:getParameters()

print(c.blue '==>' .. ' setting criterion')
criterion = nn.CrossEntropyCriterion():cl()
print(c.blue '==>' .. ' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}

function train()
  model:training()
  epoch = epoch or 1


  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then
    optimState.learningRate = optimState.learningRate / 2
  end

  print(c.blue '==>' .. ' online epoch # ' .. epoch .. ' [batchSize = ' ..
      opt.batchSize .. ']')

  local numTrainExamples = provider.trainData.data:size(1)

  local targets = torch.ClTensor(opt.batchSize)
  local indices = torch.randperm(numTrainExamples):long():split(opt.batchSize)

  -- remove last element so that all the batches have equal size
  indices[#indices] = nil
  local tic = torch.tic()

  for step, v in ipairs(indices) do
    xlua.progress(step, #indices)
    local inputs = provider.trainData.data:index(1, v)
    targets:copy(provider.trainData.labels:index(1, v))

    local feval = function(x)
      if x ~= parameters then
        parameters:copy(x)
      end
      gradParameters:zero()

      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      confusion:batchAdd(outputs, targets)
      

      return f,gradParameters
    end

    -- Perform gradient descent.
    optim.sgd(feval, parameters, optimState)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
end

function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 125
  for i=1,provider.testData.data:size(1),bs do
    local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    testLogger:plot()

    local base64im
    do
      os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/test.base64')
      if f then base64im = f:read'*all' end
    end

    local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- save model every 50 epochs
  if epoch % 50 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3))
  end

  confusion:zero()
end




for i=1, opt.max_epoch do
  train()
  test()
end


