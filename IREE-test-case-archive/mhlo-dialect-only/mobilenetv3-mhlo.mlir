module  {
  func.func @predict(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x1000xf32>{
    %210 = mhlo.constant dense<0.00784313772> : tensor<1x224x224x3xf32>
    %211 = mhlo.constant dense<-1.000000e+00> : tensor<1x224x224x3xf32>
    %212 = mhlo.constant dense<3.000000e+00> : tensor<1x112x112x16xf32>
    %213 = mhlo.constant dense<3.000000e+00> : tensor<1x1x1x16xf32>
    %214 = mhlo.constant dense<3.000000e+00> : tensor<1x28x28x96xf32>
    %215 = mhlo.constant dense<3.000000e+00> : tensor<1x14x14x96xf32>
    %216 = mhlo.constant dense<3.000000e+00> : tensor<1x1x1x96xf32>
    %217 = mhlo.constant dense<3.000000e+00> : tensor<1x14x14x240xf32>
    %218 = mhlo.constant dense<3.000000e+00> : tensor<1x1x1x240xf32>
    %219 = mhlo.constant dense<3.000000e+00> : tensor<1x14x14x120xf32>
    %220 = mhlo.constant dense<3.000000e+00> : tensor<1x1x1x120xf32>
    %221 = mhlo.constant dense<3.000000e+00> : tensor<1x14x14x144xf32>
    %222 = mhlo.constant dense<3.000000e+00> : tensor<1x1x1x144xf32>
    %223 = mhlo.constant dense<3.000000e+00> : tensor<1x14x14x288xf32>
    %224 = mhlo.constant dense<3.000000e+00> : tensor<1x7x7x288xf32>
    %225 = mhlo.constant dense<3.000000e+00> : tensor<1x1x1x288xf32>
    %226 = mhlo.constant dense<3.000000e+00> : tensor<1x1x1x576xf32>
    %227 = mhlo.constant dense<3.000000e+00> : tensor<1x7x7x576xf32>
    %228 = mhlo.constant dense<3.000000e+00> : tensor<1x1x1x1024xf32>
    %229 = mhlo.constant dense<0.166666672> : tensor<1x112x112x16xf32>
    %230 = mhlo.constant dense<0.166666672> : tensor<1x1x1x16xf32>
    %231 = mhlo.constant dense<0.166666672> : tensor<1x28x28x96xf32>
    %232 = mhlo.constant dense<0.166666672> : tensor<1x14x14x96xf32>
    %233 = mhlo.constant dense<0.166666672> : tensor<1x1x1x96xf32>
    %234 = mhlo.constant dense<0.166666672> : tensor<1x14x14x240xf32>
    %235 = mhlo.constant dense<0.166666672> : tensor<1x1x1x240xf32>
    %236 = mhlo.constant dense<0.166666672> : tensor<1x14x14x120xf32>
    %237 = mhlo.constant dense<0.166666672> : tensor<1x1x1x120xf32>
    %238 = mhlo.constant dense<0.166666672> : tensor<1x14x14x144xf32>
    %239 = mhlo.constant dense<0.166666672> : tensor<1x1x1x144xf32>
    %240 = mhlo.constant dense<0.166666672> : tensor<1x14x14x288xf32>
    %241 = mhlo.constant dense<0.166666672> : tensor<1x7x7x288xf32>
    %242 = mhlo.constant dense<0.166666672> : tensor<1x1x1x288xf32>
    %243 = mhlo.constant dense<0.166666672> : tensor<1x1x1x576xf32>
    %244 = mhlo.constant dense<0.166666672> : tensor<1x7x7x576xf32>
    %245 = mhlo.constant dense<0.166666672> : tensor<1x1x1x1024xf32>
    %246 = mhlo.constant dense<0.000000e+00> : tensor<1x56x56x16xf32>
    %247 = mhlo.constant dense<3.136000e+03> : tensor<1x16xf32>
    %248 = mhlo.constant dense<0.000000e+00> : tensor<1x1x1x8xf32>
    %249 = mhlo.constant dense<0.000000e+00> : tensor<1x56x56x72xf32>
    %250 = mhlo.constant dense<0.000000e+00> : tensor<1x28x28x72xf32>
    %251 = mhlo.constant dense<0.000000e+00> : tensor<1x28x28x88xf32>
    %252 = mhlo.constant dense<1.960000e+02> : tensor<1x96xf32>
    %253 = mhlo.constant dense<0.000000e+00> : tensor<1x1x1x24xf32>
    %254 = mhlo.constant dense<1.960000e+02> : tensor<1x240xf32>
    %255 = mhlo.constant dense<0.000000e+00> : tensor<1x1x1x64xf32>
    %256 = mhlo.constant dense<1.960000e+02> : tensor<1x120xf32>
    %257 = mhlo.constant dense<0.000000e+00> : tensor<1x1x1x32xf32>
    %258 = mhlo.constant dense<1.960000e+02> : tensor<1x144xf32>
    %259 = mhlo.constant dense<0.000000e+00> : tensor<1x1x1x40xf32>
    %260 = mhlo.constant dense<4.900000e+01> : tensor<1x288xf32>
    %261 = mhlo.constant dense<0.000000e+00> : tensor<1x1x1x72xf32>
    %262 = mhlo.constant dense<0.000000e+00> : tensor<1x1x1x144xf32>
    %263 = mhlo.constant dense<4.900000e+01> : tensor<1x576xf32>
    %264 = mhlo.constant dense<6.000000e+00> : tensor<f32>
    %265 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %266 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %267 = arith.constant dense<0.00485436898> : tensor<576xf32>
    %268 = arith.constant dense<0.00487804879> : tensor<576xf32>
    %269 = arith.constant dense<0.00490196096> : tensor<576xf32>
    %270 = arith.constant dense<0.00492610829> : tensor<576xf32>
    %271 = arith.constant dense<0.00495049497> : tensor<1x1x96x576xf32>
    %272 = arith.constant dense<0.00480769249> : tensor<1024xf32>
    %273 = arith.constant dense<0.00483091781> : tensor<1x1x576x1024xf32>
    %274 = arith.constant dense<2.000000e-01> : tensor<16xf32>
    %275 = arith.constant dense<2.500000e-01> : tensor<16xf32>
    %276 = arith.constant dense<0.333333343> : tensor<16xf32>
    %277 = arith.constant dense<5.000000e-01> : tensor<16xf32>
    %278 = arith.constant dense<1.000000e+00> : tensor<3x3x3x16xf32>
    %279 = arith.constant dense<0.00520833349> : tensor<576xf32>
    %280 = arith.constant dense<0.00523560215> : tensor<576xf32>
    %281 = arith.constant dense<0.00526315812> : tensor<576xf32>
    %282 = arith.constant dense<0.00529100513> : tensor<576xf32>
    %283 = arith.constant dense<0.00531914877> : tensor<5x5x576x1xf32>
    %284 = arith.constant dense<0.00534759369> : tensor<576xf32>
    %285 = arith.constant dense<0.00537634408> : tensor<576xf32>
    %286 = arith.constant dense<0.00540540554> : tensor<576xf32>
    %287 = arith.constant dense<0.00543478271> : tensor<576xf32>
    %288 = arith.constant dense<0.00546448072> : tensor<1x1x96x576xf32>
    %289 = arith.constant dense<0.00497512426> : tensor<96xf32>
    %290 = arith.constant dense<5.000000e-03> : tensor<96xf32>
    %291 = arith.constant dense<0.00502512557> : tensor<96xf32>
    %292 = arith.constant dense<0.00505050505> : tensor<96xf32>
    %293 = arith.constant dense<0.00507614203> : tensor<1x1x576x96xf32>
    %294 = arith.constant dense<0.00510204071> : tensor<576xf32>
    %295 = arith.constant dense<0.00512820529> : tensor<1x1x144x576xf32>
    %296 = arith.constant dense<0.00515463902> : tensor<144xf32>
    %297 = arith.constant dense<0.00518134702> : tensor<1x1x576x144xf32>
    %298 = arith.constant dense<0.0344827585> : tensor<72xf32>
    %299 = arith.constant dense<0.0357142873> : tensor<72xf32>
    %300 = arith.constant dense<0.0370370373> : tensor<72xf32>
    %301 = arith.constant dense<0.0384615399> : tensor<72xf32>
    %302 = arith.constant dense<4.000000e-02> : tensor<3x3x72x1xf32>
    %303 = arith.constant dense<0.0416666679> : tensor<72xf32>
    %304 = arith.constant dense<0.0434782617> : tensor<72xf32>
    %305 = arith.constant dense<0.0454545468> : tensor<72xf32>
    %306 = arith.constant dense<0.0476190485> : tensor<72xf32>
    %307 = arith.constant dense<5.000000e-02> : tensor<1x1x16x72xf32>
    %308 = arith.constant dense<0.0294117648> : tensor<24xf32>
    %309 = arith.constant dense<0.0303030312> : tensor<24xf32>
    %310 = arith.constant dense<3.125000e-02> : tensor<24xf32>
    %311 = arith.constant dense<0.0322580636> : tensor<24xf32>
    %312 = arith.constant dense<0.0333333351> : tensor<1x1x72x24xf32>
    %313 = arith.constant dense<0.0227272734> : tensor<88xf32>
    %314 = arith.constant dense<0.0232558139> : tensor<88xf32>
    %315 = arith.constant dense<0.0238095243> : tensor<88xf32>
    %316 = arith.constant dense<0.024390243> : tensor<88xf32>
    %317 = arith.constant dense<2.500000e-02> : tensor<3x3x88x1xf32>
    %318 = arith.constant dense<0.025641026> : tensor<88xf32>
    %319 = arith.constant dense<0.0263157897> : tensor<88xf32>
    %320 = arith.constant dense<0.0270270277> : tensor<88xf32>
    %321 = arith.constant dense<0.027777778> : tensor<88xf32>
    %322 = arith.constant dense<0.0285714287> : tensor<1x1x24x88xf32>
    %323 = arith.constant dense<0.0204081628> : tensor<24xf32>
    %324 = arith.constant dense<0.020833334> : tensor<24xf32>
    %325 = arith.constant dense<0.0212765951> : tensor<24xf32>
    %326 = arith.constant dense<0.0217391308> : tensor<24xf32>
    %327 = arith.constant dense<0.0222222228> : tensor<1x1x88x24xf32>
    %328 = arith.constant dense<0.0169491526> : tensor<96xf32>
    %329 = arith.constant dense<0.0172413792> : tensor<96xf32>
    %330 = arith.constant dense<0.0175438598> : tensor<96xf32>
    %331 = arith.constant dense<0.0178571437> : tensor<96xf32>
    %332 = arith.constant dense<0.0181818176> : tensor<5x5x96x1xf32>
    %333 = arith.constant dense<0.0185185187> : tensor<96xf32>
    %334 = arith.constant dense<0.0188679248> : tensor<96xf32>
    %335 = arith.constant dense<0.0192307699> : tensor<96xf32>
    %336 = arith.constant dense<0.0196078438> : tensor<96xf32>
    %337 = arith.constant dense<2.000000e-02> : tensor<1x1x24x96xf32>
    %338 = arith.constant dense<0.0147058824> : tensor<40xf32>
    %339 = arith.constant dense<0.0149253728> : tensor<40xf32>
    %340 = arith.constant dense<0.0151515156> : tensor<40xf32>
    %341 = arith.constant dense<0.0153846154> : tensor<40xf32>
    %342 = arith.constant dense<1.562500e-02> : tensor<1x1x96x40xf32>
    %343 = arith.constant dense<0.0158730168> : tensor<96xf32>
    %344 = arith.constant dense<0.0161290318> : tensor<1x1x24x96xf32>
    %345 = arith.constant dense<0.0163934417> : tensor<24xf32>
    %346 = arith.constant dense<0.0166666675> : tensor<1x1x96x24xf32>
    %347 = arith.constant dense<0.012820513> : tensor<240xf32>
    %348 = arith.constant dense<0.012987013> : tensor<240xf32>
    %349 = arith.constant dense<0.0131578948> : tensor<240xf32>
    %350 = arith.constant dense<0.0133333337> : tensor<240xf32>
    %351 = arith.constant dense<0.0135135138> : tensor<5x5x240x1xf32>
    %352 = arith.constant dense<0.01369863> : tensor<240xf32>
    %353 = arith.constant dense<0.013888889> : tensor<240xf32>
    %354 = arith.constant dense<0.0140845068> : tensor<240xf32>
    %355 = arith.constant dense<0.0142857144> : tensor<240xf32>
    %356 = arith.constant dense<0.0144927539> : tensor<1x1x40x240xf32>
    %357 = arith.constant dense<0.0114942528> : tensor<40xf32>
    %358 = arith.constant dense<0.0116279069> : tensor<40xf32>
    %359 = arith.constant dense<0.0117647061> : tensor<40xf32>
    %360 = arith.constant dense<0.0119047621> : tensor<40xf32>
    %361 = arith.constant dense<0.0120481923> : tensor<1x1x240x40xf32>
    %362 = arith.constant dense<0.0121951215> : tensor<240xf32>
    %363 = arith.constant dense<0.0123456791> : tensor<1x1x64x240xf32>
    %364 = arith.constant dense<1.250000e-02> : tensor<64xf32>
    %365 = arith.constant dense<0.0126582282> : tensor<1x1x240x64xf32>
    %366 = arith.constant dense<0.010309278> : tensor<240xf32>
    %367 = arith.constant dense<0.010416667> : tensor<240xf32>
    %368 = arith.constant dense<0.0105263162> : tensor<240xf32>
    %369 = arith.constant dense<0.0106382975> : tensor<240xf32>
    %370 = arith.constant dense<0.0107526882> : tensor<5x5x240x1xf32>
    %371 = arith.constant dense<0.0108695654> : tensor<240xf32>
    %372 = arith.constant dense<0.0109890113> : tensor<240xf32>
    %373 = arith.constant dense<0.0111111114> : tensor<240xf32>
    %374 = arith.constant dense<0.0112359552> : tensor<240xf32>
    %375 = arith.constant dense<0.0113636367> : tensor<1x1x40x240xf32>
    %376 = arith.constant dense<0.0094339624> : tensor<40xf32>
    %377 = arith.constant dense<9.523810e-03> : tensor<40xf32>
    %378 = arith.constant dense<0.00961538497> : tensor<40xf32>
    %379 = arith.constant dense<0.00970873795> : tensor<40xf32>
    %380 = arith.constant dense<0.00980392192> : tensor<1x1x240x40xf32>
    %381 = arith.constant dense<9.900990e-03> : tensor<240xf32>
    %382 = arith.constant dense<0.00999999977> : tensor<1x1x64x240xf32>
    %383 = arith.constant dense<0.0101010101> : tensor<64xf32>
    %384 = arith.constant dense<0.0102040814> : tensor<1x1x240x64xf32>
    %385 = arith.constant dense<8.620690e-03> : tensor<120xf32>
    %386 = arith.constant dense<0.00869565178> : tensor<120xf32>
    %387 = arith.constant dense<0.00877192988> : tensor<120xf32>
    %388 = arith.constant dense<0.00884955748> : tensor<120xf32>
    %389 = arith.constant dense<0.00892857183> : tensor<5x5x120x1xf32>
    %390 = arith.constant dense<0.00900900922> : tensor<120xf32>
    %391 = arith.constant dense<0.0090909088> : tensor<120xf32>
    %392 = arith.constant dense<0.00917431153> : tensor<120xf32>
    %393 = arith.constant dense<0.00925925932> : tensor<120xf32>
    %394 = arith.constant dense<0.00934579409> : tensor<1x1x40x120xf32>
    %395 = arith.constant dense<8.000000e-03> : tensor<48xf32>
    %396 = arith.constant dense<0.00806451589> : tensor<48xf32>
    %397 = arith.constant dense<0.008130081> : tensor<48xf32>
    %398 = arith.constant dense<0.00819672085> : tensor<48xf32>
    %399 = arith.constant dense<0.00826446246> : tensor<1x1x120x48xf32>
    %400 = arith.constant dense<0.00833333377> : tensor<120xf32>
    %401 = arith.constant dense<0.00840336177> : tensor<1x1x32x120xf32>
    %402 = arith.constant dense<0.00847457629> : tensor<32xf32>
    %403 = arith.constant dense<0.00854700897> : tensor<1x1x120x32xf32>
    %404 = arith.constant dense<0.00740740728> : tensor<144xf32>
    %405 = arith.constant dense<0.00746268639> : tensor<144xf32>
    %406 = arith.constant dense<0.00751879718> : tensor<144xf32>
    %407 = arith.constant dense<0.0075757578> : tensor<144xf32>
    %408 = arith.constant dense<0.00763358781> : tensor<5x5x144x1xf32>
    %409 = arith.constant dense<0.0076923077> : tensor<144xf32>
    %410 = arith.constant dense<0.00775193795> : tensor<144xf32>
    %411 = arith.constant dense<7.812500e-03> : tensor<144xf32>
    %412 = arith.constant dense<0.00787401571> : tensor<144xf32>
    %413 = arith.constant dense<0.00793650839> : tensor<1x1x48x144xf32>
    %414 = arith.constant dense<0.0069444445> : tensor<48xf32>
    %415 = arith.constant dense<0.00699300691> : tensor<48xf32>
    %416 = arith.constant dense<0.00704225338> : tensor<48xf32>
    %417 = arith.constant dense<0.00709219835> : tensor<48xf32>
    %418 = arith.constant dense<0.00714285718> : tensor<1x1x144x48xf32>
    %419 = arith.constant dense<0.00719424477> : tensor<144xf32>
    %420 = arith.constant dense<0.00724637694> : tensor<1x1x40x144xf32>
    %421 = arith.constant dense<7.299270e-03> : tensor<40xf32>
    %422 = arith.constant dense<0.0073529412> : tensor<1x1x144x40xf32>
    %423 = arith.constant dense<0.00649350649> : tensor<288xf32>
    %424 = arith.constant dense<0.00653594779> : tensor<288xf32>
    %425 = arith.constant dense<0.00657894742> : tensor<288xf32>
    %426 = arith.constant dense<0.00662251655> : tensor<288xf32>
    %427 = arith.constant dense<0.00666666683> : tensor<5x5x288x1xf32>
    %428 = arith.constant dense<0.00671140943> : tensor<288xf32>
    %429 = arith.constant dense<0.00675675692> : tensor<288xf32>
    %430 = arith.constant dense<0.00680272094> : tensor<288xf32>
    %431 = arith.constant dense<0.00684931502> : tensor<288xf32>
    %432 = arith.constant dense<0.0068965517> : tensor<1x1x48x288xf32>
    %433 = arith.constant dense<0.00613496918> : tensor<96xf32>
    %434 = arith.constant dense<0.00617283955> : tensor<96xf32>
    %435 = arith.constant dense<0.00621118024> : tensor<96xf32>
    %436 = arith.constant dense<6.250000e-03> : tensor<96xf32>
    %437 = arith.constant dense<0.00628930796> : tensor<1x1x288x96xf32>
    %438 = arith.constant dense<0.00632911408> : tensor<288xf32>
    %439 = arith.constant dense<0.00636942684> : tensor<1x1x72x288xf32>
    %440 = arith.constant dense<0.00641025649> : tensor<72xf32>
    %441 = arith.constant dense<0.0064516128> : tensor<1x1x288x72xf32>
    %442 = arith.constant dense<0.00578034669> : tensor<576xf32>
    %443 = arith.constant dense<0.00581395347> : tensor<576xf32>
    %444 = arith.constant dense<0.00584795326> : tensor<576xf32>
    %445 = arith.constant dense<0.00588235306> : tensor<576xf32>
    %446 = arith.constant dense<5.917160e-03> : tensor<5x5x576x1xf32>
    %447 = arith.constant dense<0.00595238106> : tensor<576xf32>
    %448 = arith.constant dense<0.00598802418> : tensor<576xf32>
    %449 = arith.constant dense<0.00602409616> : tensor<576xf32>
    %450 = arith.constant dense<0.00606060587> : tensor<576xf32>
    %451 = arith.constant dense<0.00609756075> : tensor<1x1x96x576xf32>
    %452 = arith.constant dense<0.00549450563> : tensor<96xf32>
    %453 = arith.constant dense<0.00552486209> : tensor<96xf32>
    %454 = arith.constant dense<0.00555555569> : tensor<96xf32>
    %455 = arith.constant dense<0.00558659201> : tensor<96xf32>
    %456 = arith.constant dense<0.00561797759> : tensor<1x1x576x96xf32>
    %457 = arith.constant dense<0.00564971752> : tensor<576xf32>
    %458 = arith.constant dense<0.00568181835> : tensor<1x1x144x576xf32>
    %459 = arith.constant dense<0.00571428565> : tensor<144xf32>
    %460 = arith.constant dense<0.00574712642> : tensor<1x1x576x144xf32>
    %461 = arith.constant dense<1.000000e-01> : tensor<16xf32>
    %462 = arith.constant dense<0.111111112> : tensor<16xf32>
    %463 = arith.constant dense<1.250000e-01> : tensor<16xf32>
    %464 = arith.constant dense<0.142857149> : tensor<16xf32>
    %465 = arith.constant dense<0.166666672> : tensor<3x3x16x1xf32>
    %466 = arith.constant dense<0.0526315793> : tensor<16xf32>
    %467 = arith.constant dense<0.055555556> : tensor<16xf32>
    %468 = arith.constant dense<0.0588235296> : tensor<16xf32>
    %469 = arith.constant dense<6.250000e-02> : tensor<16xf32>
    %470 = arith.constant dense<0.0666666701> : tensor<1x1x16x16xf32>
    %471 = arith.constant dense<0.0714285746> : tensor<16xf32>
    %472 = arith.constant dense<0.0769230798> : tensor<1x1x8x16xf32>
    %473 = arith.constant dense<0.0833333358> : tensor<8xf32>
    %474 = arith.constant dense<0.0909090936> : tensor<1x1x16x8xf32>
    %475 = arith.constant dense<0.00476190494> : tensor<1000xf32>
    %476 = arith.constant dense<0.00478468882> : tensor<1x1x1024x1000xf32>
    %477 = mhlo.multiply %arg0, %210 : tensor<1x224x224x3xf32>
    %478 = mhlo.add %477, %211 : tensor<1x224x224x3xf32>
    %479 = "mhlo.convolution"(%478, %278) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x224x224x3xf32>, tensor<3x3x3x16xf32>) -> tensor<1x112x112x16xf32>
    %480 = "mhlo.batch_norm_inference"(%479, %277, %276, %275, %274) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x112x112x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1x112x112x16xf32>
    %481 = mhlo.add %480, %212 : tensor<1x112x112x16xf32>
    %482 = "mhlo.clamp"(%266, %481, %264) : (tensor<f32>, tensor<1x112x112x16xf32>, tensor<f32>) -> tensor<1x112x112x16xf32>
    %483 = mhlo.multiply %482, %229 : tensor<1x112x112x16xf32>
    %484 = mhlo.multiply %483, %480 : tensor<1x112x112x16xf32>
    %485 = "mhlo.pad"(%484, %266) {edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>, edge_padding_low = dense<0> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x112x112x16xf32>, tensor<f32>) -> tensor<1x113x113x16xf32>
    %486 = "mhlo.reshape"(%465) : (tensor<3x3x16x1xf32>) -> tensor<3x3x1x16xf32>
    %487 = "mhlo.convolution"(%485, %486) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 16 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x113x113x16xf32>, tensor<3x3x1x16xf32>) -> tensor<1x56x56x16xf32>
    %488 = "mhlo.batch_norm_inference"(%487, %464, %463, %462, %461) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1x56x56x16xf32>
    %489 = mhlo.maximum %488, %246 : tensor<1x56x56x16xf32>
    %490 = "mhlo.reduce"(%489, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x56x56x16xf32>, tensor<f32>) -> tensor<1x16xf32>
    %491 = mhlo.divide %490, %247 : tensor<1x16xf32>
    %492 = "mhlo.reshape"(%491) : (tensor<1x16xf32>) -> tensor<1x1x1x16xf32>
    %493 = "mhlo.convolution"(%492, %474) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x16xf32>, tensor<1x1x16x8xf32>) -> tensor<1x1x1x8xf32>
    %494 = "mhlo.broadcast_in_dim"(%473) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<8xf32>) -> tensor<1x1x1x8xf32>
    %495 = mhlo.add %493, %494 : tensor<1x1x1x8xf32>
    %496 = mhlo.maximum %495, %248 : tensor<1x1x1x8xf32>
    %497 = "mhlo.convolution"(%496, %472) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x8xf32>, tensor<1x1x8x16xf32>) -> tensor<1x1x1x16xf32>
    %498 = "mhlo.broadcast_in_dim"(%471) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %499 = mhlo.add %497, %498 : tensor<1x1x1x16xf32>
    %500 = mhlo.add %499, %213 : tensor<1x1x1x16xf32>
    %501 = "mhlo.clamp"(%266, %500, %264) : (tensor<f32>, tensor<1x1x1x16xf32>, tensor<f32>) -> tensor<1x1x1x16xf32>
    %502 = mhlo.multiply %501, %230 : tensor<1x1x1x16xf32>
    %503 = "mhlo.broadcast_in_dim"(%502) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x16xf32>) -> tensor<1x56x56x16xf32>
    %504 = mhlo.multiply %489, %503 : tensor<1x56x56x16xf32>
    %505 = "mhlo.convolution"(%504, %470) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x16xf32>, tensor<1x1x16x16xf32>) -> tensor<1x56x56x16xf32>
    %506 = "mhlo.batch_norm_inference"(%505, %469, %468, %467, %466) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<1x56x56x16xf32>
    %507 = "mhlo.convolution"(%506, %307) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x56x56x16xf32>, tensor<1x1x16x72xf32>) -> tensor<1x56x56x72xf32>
    %508 = "mhlo.batch_norm_inference"(%507, %306, %305, %304, %303) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x56x56x72xf32>, tensor<72xf32>, tensor<72xf32>, tensor<72xf32>, tensor<72xf32>) -> tensor<1x56x56x72xf32>
    %509 = mhlo.maximum %508, %249 : tensor<1x56x56x72xf32>
    %510 = "mhlo.pad"(%509, %266) {edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>, edge_padding_low = dense<0> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x56x56x72xf32>, tensor<f32>) -> tensor<1x57x57x72xf32>
    %511 = "mhlo.reshape"(%302) : (tensor<3x3x72x1xf32>) -> tensor<3x3x1x72xf32>
    %512 = "mhlo.convolution"(%510, %511) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 72 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x57x57x72xf32>, tensor<3x3x1x72xf32>) -> tensor<1x28x28x72xf32>
    %513 = "mhlo.batch_norm_inference"(%512, %301, %300, %299, %298) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x72xf32>, tensor<72xf32>, tensor<72xf32>, tensor<72xf32>, tensor<72xf32>) -> tensor<1x28x28x72xf32>
    %514 = mhlo.maximum %513, %250 : tensor<1x28x28x72xf32>
    %515 = "mhlo.convolution"(%514, %312) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x72xf32>, tensor<1x1x72x24xf32>) -> tensor<1x28x28x24xf32>
    %516 = "mhlo.batch_norm_inference"(%515, %311, %310, %309, %308) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>) -> tensor<1x28x28x24xf32>
    %517 = "mhlo.convolution"(%516, %322) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x24xf32>, tensor<1x1x24x88xf32>) -> tensor<1x28x28x88xf32>
    %518 = "mhlo.batch_norm_inference"(%517, %321, %320, %319, %318) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x88xf32>, tensor<88xf32>, tensor<88xf32>, tensor<88xf32>, tensor<88xf32>) -> tensor<1x28x28x88xf32>
    %519 = mhlo.maximum %518, %251 : tensor<1x28x28x88xf32>
    %520 = "mhlo.reshape"(%317) : (tensor<3x3x88x1xf32>) -> tensor<3x3x1x88xf32>
    %521 = "mhlo.convolution"(%519, %520) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 88 : i64, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x88xf32>, tensor<3x3x1x88xf32>) -> tensor<1x28x28x88xf32>
    %522 = "mhlo.batch_norm_inference"(%521, %316, %315, %314, %313) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x88xf32>, tensor<88xf32>, tensor<88xf32>, tensor<88xf32>, tensor<88xf32>) -> tensor<1x28x28x88xf32>
    %523 = mhlo.maximum %522, %251 : tensor<1x28x28x88xf32>
    %524 = "mhlo.convolution"(%523, %327) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x88xf32>, tensor<1x1x88x24xf32>) -> tensor<1x28x28x24xf32>
    %525 = "mhlo.batch_norm_inference"(%524, %326, %325, %324, %323) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<24xf32>) -> tensor<1x28x28x24xf32>
    %526 = mhlo.add %516, %525 : tensor<1x28x28x24xf32>
    %527 = "mhlo.convolution"(%526, %337) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x28x28x24xf32>, tensor<1x1x24x96xf32>) -> tensor<1x28x28x96xf32>
    %528 = "mhlo.batch_norm_inference"(%527, %336, %335, %334, %333) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x28x28x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x28x28x96xf32>
    %529 = mhlo.add %528, %214 : tensor<1x28x28x96xf32>
    %530 = "mhlo.clamp"(%266, %529, %264) : (tensor<f32>, tensor<1x28x28x96xf32>, tensor<f32>) -> tensor<1x28x28x96xf32>
    %531 = mhlo.multiply %530, %231 : tensor<1x28x28x96xf32>
    %532 = mhlo.multiply %531, %528 : tensor<1x28x28x96xf32>
    %533 = "mhlo.pad"(%532, %266) {edge_padding_high = dense<[0, 2, 2, 0]> : tensor<4xi64>, edge_padding_low = dense<[0, 1, 1, 0]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x28x28x96xf32>, tensor<f32>) -> tensor<1x31x31x96xf32>
    %534 = "mhlo.reshape"(%332) : (tensor<5x5x96x1xf32>) -> tensor<5x5x1x96xf32>
    %535 = "mhlo.convolution"(%533, %534) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 96 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x31x31x96xf32>, tensor<5x5x1x96xf32>) -> tensor<1x14x14x96xf32>
    %536 = "mhlo.batch_norm_inference"(%535, %331, %330, %329, %328) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x14x14x96xf32>
    %537 = mhlo.add %536, %215 : tensor<1x14x14x96xf32>
    %538 = "mhlo.clamp"(%266, %537, %264) : (tensor<f32>, tensor<1x14x14x96xf32>, tensor<f32>) -> tensor<1x14x14x96xf32>
    %539 = mhlo.multiply %538, %232 : tensor<1x14x14x96xf32>
    %540 = mhlo.multiply %539, %536 : tensor<1x14x14x96xf32>
    %541 = "mhlo.reduce"(%540, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x14x14x96xf32>, tensor<f32>) -> tensor<1x96xf32>
    %542 = mhlo.divide %541, %252 : tensor<1x96xf32>
    %543 = "mhlo.reshape"(%542) : (tensor<1x96xf32>) -> tensor<1x1x1x96xf32>
    %544 = "mhlo.convolution"(%543, %346) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x96xf32>, tensor<1x1x96x24xf32>) -> tensor<1x1x1x24xf32>
    %545 = "mhlo.broadcast_in_dim"(%345) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<24xf32>) -> tensor<1x1x1x24xf32>
    %546 = mhlo.add %544, %545 : tensor<1x1x1x24xf32>
    %547 = mhlo.maximum %546, %253 : tensor<1x1x1x24xf32>
    %548 = "mhlo.convolution"(%547, %344) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x24xf32>, tensor<1x1x24x96xf32>) -> tensor<1x1x1x96xf32>
    %549 = "mhlo.broadcast_in_dim"(%343) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<96xf32>) -> tensor<1x1x1x96xf32>
    %550 = mhlo.add %548, %549 : tensor<1x1x1x96xf32>
    %551 = mhlo.add %550, %216 : tensor<1x1x1x96xf32>
    %552 = "mhlo.clamp"(%266, %551, %264) : (tensor<f32>, tensor<1x1x1x96xf32>, tensor<f32>) -> tensor<1x1x1x96xf32>
    %553 = mhlo.multiply %552, %233 : tensor<1x1x1x96xf32>
    %554 = "mhlo.broadcast_in_dim"(%553) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x96xf32>) -> tensor<1x14x14x96xf32>
    %555 = mhlo.multiply %540, %554 : tensor<1x14x14x96xf32>
    %556 = "mhlo.convolution"(%555, %342) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x96xf32>, tensor<1x1x96x40xf32>) -> tensor<1x14x14x40xf32>
    %557 = "mhlo.batch_norm_inference"(%556, %341, %340, %339, %338) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>) -> tensor<1x14x14x40xf32>
    %558 = "mhlo.convolution"(%557, %356) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x40xf32>, tensor<1x1x40x240xf32>) -> tensor<1x14x14x240xf32>
    %559 = "mhlo.batch_norm_inference"(%558, %355, %354, %353, %352) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>) -> tensor<1x14x14x240xf32>
    %560 = mhlo.add %559, %217 : tensor<1x14x14x240xf32>
    %561 = "mhlo.clamp"(%266, %560, %264) : (tensor<f32>, tensor<1x14x14x240xf32>, tensor<f32>) -> tensor<1x14x14x240xf32>
    %562 = mhlo.multiply %561, %234 : tensor<1x14x14x240xf32>
    %563 = mhlo.multiply %562, %559 : tensor<1x14x14x240xf32>
    %564 = "mhlo.reshape"(%351) : (tensor<5x5x240x1xf32>) -> tensor<5x5x1x240xf32>
    %565 = "mhlo.convolution"(%563, %564) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 240 : i64, padding = dense<2> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x240xf32>, tensor<5x5x1x240xf32>) -> tensor<1x14x14x240xf32>
    %566 = "mhlo.batch_norm_inference"(%565, %350, %349, %348, %347) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>) -> tensor<1x14x14x240xf32>
    %567 = mhlo.add %566, %217 : tensor<1x14x14x240xf32>
    %568 = "mhlo.clamp"(%266, %567, %264) : (tensor<f32>, tensor<1x14x14x240xf32>, tensor<f32>) -> tensor<1x14x14x240xf32>
    %569 = mhlo.multiply %568, %234 : tensor<1x14x14x240xf32>
    %570 = mhlo.multiply %569, %566 : tensor<1x14x14x240xf32>
    %571 = "mhlo.reduce"(%570, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x14x14x240xf32>, tensor<f32>) -> tensor<1x240xf32>
    %572 = mhlo.divide %571, %254 : tensor<1x240xf32>
    %573 = "mhlo.reshape"(%572) : (tensor<1x240xf32>) -> tensor<1x1x1x240xf32>
    %574 = "mhlo.convolution"(%573, %365) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x240xf32>, tensor<1x1x240x64xf32>) -> tensor<1x1x1x64xf32>
    %575 = "mhlo.broadcast_in_dim"(%364) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %576 = mhlo.add %574, %575 : tensor<1x1x1x64xf32>
    %577 = mhlo.maximum %576, %255 : tensor<1x1x1x64xf32>
    %578 = "mhlo.convolution"(%577, %363) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x64xf32>, tensor<1x1x64x240xf32>) -> tensor<1x1x1x240xf32>
    %579 = "mhlo.broadcast_in_dim"(%362) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<240xf32>) -> tensor<1x1x1x240xf32>
    %580 = mhlo.add %578, %579 : tensor<1x1x1x240xf32>
    %581 = mhlo.add %580, %218 : tensor<1x1x1x240xf32>
    %582 = "mhlo.clamp"(%266, %581, %264) : (tensor<f32>, tensor<1x1x1x240xf32>, tensor<f32>) -> tensor<1x1x1x240xf32>
    %583 = mhlo.multiply %582, %235 : tensor<1x1x1x240xf32>
    %584 = "mhlo.broadcast_in_dim"(%583) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x240xf32>) -> tensor<1x14x14x240xf32>
    %585 = mhlo.multiply %570, %584 : tensor<1x14x14x240xf32>
    %586 = "mhlo.convolution"(%585, %361) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x240xf32>, tensor<1x1x240x40xf32>) -> tensor<1x14x14x40xf32>
    %587 = "mhlo.batch_norm_inference"(%586, %360, %359, %358, %357) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>) -> tensor<1x14x14x40xf32>
    %588 = mhlo.add %557, %587 : tensor<1x14x14x40xf32>
    %589 = "mhlo.convolution"(%588, %375) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x40xf32>, tensor<1x1x40x240xf32>) -> tensor<1x14x14x240xf32>
    %590 = "mhlo.batch_norm_inference"(%589, %374, %373, %372, %371) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>) -> tensor<1x14x14x240xf32>
    %591 = mhlo.add %590, %217 : tensor<1x14x14x240xf32>
    %592 = "mhlo.clamp"(%266, %591, %264) : (tensor<f32>, tensor<1x14x14x240xf32>, tensor<f32>) -> tensor<1x14x14x240xf32>
    %593 = mhlo.multiply %592, %234 : tensor<1x14x14x240xf32>
    %594 = mhlo.multiply %593, %590 : tensor<1x14x14x240xf32>
    %595 = "mhlo.reshape"(%370) : (tensor<5x5x240x1xf32>) -> tensor<5x5x1x240xf32>
    %596 = "mhlo.convolution"(%594, %595) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 240 : i64, padding = dense<2> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x240xf32>, tensor<5x5x1x240xf32>) -> tensor<1x14x14x240xf32>
    %597 = "mhlo.batch_norm_inference"(%596, %369, %368, %367, %366) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>, tensor<240xf32>) -> tensor<1x14x14x240xf32>
    %598 = mhlo.add %597, %217 : tensor<1x14x14x240xf32>
    %599 = "mhlo.clamp"(%266, %598, %264) : (tensor<f32>, tensor<1x14x14x240xf32>, tensor<f32>) -> tensor<1x14x14x240xf32>
    %600 = mhlo.multiply %599, %234 : tensor<1x14x14x240xf32>
    %601 = mhlo.multiply %600, %597 : tensor<1x14x14x240xf32>
    %602 = "mhlo.reduce"(%601, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x14x14x240xf32>, tensor<f32>) -> tensor<1x240xf32>
    %603 = mhlo.divide %602, %254 : tensor<1x240xf32>
    %604 = "mhlo.reshape"(%603) : (tensor<1x240xf32>) -> tensor<1x1x1x240xf32>
    %605 = "mhlo.convolution"(%604, %384) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x240xf32>, tensor<1x1x240x64xf32>) -> tensor<1x1x1x64xf32>
    %606 = "mhlo.broadcast_in_dim"(%383) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %607 = mhlo.add %605, %606 : tensor<1x1x1x64xf32>
    %608 = mhlo.maximum %607, %255 : tensor<1x1x1x64xf32>
    %609 = "mhlo.convolution"(%608, %382) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x64xf32>, tensor<1x1x64x240xf32>) -> tensor<1x1x1x240xf32>
    %610 = "mhlo.broadcast_in_dim"(%381) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<240xf32>) -> tensor<1x1x1x240xf32>
    %611 = mhlo.add %609, %610 : tensor<1x1x1x240xf32>
    %612 = mhlo.add %611, %218 : tensor<1x1x1x240xf32>
    %613 = "mhlo.clamp"(%266, %612, %264) : (tensor<f32>, tensor<1x1x1x240xf32>, tensor<f32>) -> tensor<1x1x1x240xf32>
    %614 = mhlo.multiply %613, %235 : tensor<1x1x1x240xf32>
    %615 = "mhlo.broadcast_in_dim"(%614) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x240xf32>) -> tensor<1x14x14x240xf32>
    %616 = mhlo.multiply %601, %615 : tensor<1x14x14x240xf32>
    %617 = "mhlo.convolution"(%616, %380) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x240xf32>, tensor<1x1x240x40xf32>) -> tensor<1x14x14x40xf32>
    %618 = "mhlo.batch_norm_inference"(%617, %379, %378, %377, %376) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>, tensor<40xf32>) -> tensor<1x14x14x40xf32>
    %619 = mhlo.add %588, %618 : tensor<1x14x14x40xf32>
    %620 = "mhlo.convolution"(%619, %394) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x40xf32>, tensor<1x1x40x120xf32>) -> tensor<1x14x14x120xf32>
    %621 = "mhlo.batch_norm_inference"(%620, %393, %392, %391, %390) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x120xf32>, tensor<120xf32>, tensor<120xf32>, tensor<120xf32>, tensor<120xf32>) -> tensor<1x14x14x120xf32>
    %622 = mhlo.add %621, %219 : tensor<1x14x14x120xf32>
    %623 = "mhlo.clamp"(%266, %622, %264) : (tensor<f32>, tensor<1x14x14x120xf32>, tensor<f32>) -> tensor<1x14x14x120xf32>
    %624 = mhlo.multiply %623, %236 : tensor<1x14x14x120xf32>
    %625 = mhlo.multiply %624, %621 : tensor<1x14x14x120xf32>
    %626 = "mhlo.reshape"(%389) : (tensor<5x5x120x1xf32>) -> tensor<5x5x1x120xf32>
    %627 = "mhlo.convolution"(%625, %626) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 120 : i64, padding = dense<2> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x120xf32>, tensor<5x5x1x120xf32>) -> tensor<1x14x14x120xf32>
    %628 = "mhlo.batch_norm_inference"(%627, %388, %387, %386, %385) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x120xf32>, tensor<120xf32>, tensor<120xf32>, tensor<120xf32>, tensor<120xf32>) -> tensor<1x14x14x120xf32>
    %629 = mhlo.add %628, %219 : tensor<1x14x14x120xf32>
    %630 = "mhlo.clamp"(%266, %629, %264) : (tensor<f32>, tensor<1x14x14x120xf32>, tensor<f32>) -> tensor<1x14x14x120xf32>
    %631 = mhlo.multiply %630, %236 : tensor<1x14x14x120xf32>
    %632 = mhlo.multiply %631, %628 : tensor<1x14x14x120xf32>
    %633 = "mhlo.reduce"(%632, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x14x14x120xf32>, tensor<f32>) -> tensor<1x120xf32>
    %634 = mhlo.divide %633, %256 : tensor<1x120xf32>
    %635 = "mhlo.reshape"(%634) : (tensor<1x120xf32>) -> tensor<1x1x1x120xf32>
    %636 = "mhlo.convolution"(%635, %403) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x120xf32>, tensor<1x1x120x32xf32>) -> tensor<1x1x1x32xf32>
    %637 = "mhlo.broadcast_in_dim"(%402) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<1x1x1x32xf32>
    %638 = mhlo.add %636, %637 : tensor<1x1x1x32xf32>
    %639 = mhlo.maximum %638, %257 : tensor<1x1x1x32xf32>
    %640 = "mhlo.convolution"(%639, %401) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x32xf32>, tensor<1x1x32x120xf32>) -> tensor<1x1x1x120xf32>
    %641 = "mhlo.broadcast_in_dim"(%400) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<120xf32>) -> tensor<1x1x1x120xf32>
    %642 = mhlo.add %640, %641 : tensor<1x1x1x120xf32>
    %643 = mhlo.add %642, %220 : tensor<1x1x1x120xf32>
    %644 = "mhlo.clamp"(%266, %643, %264) : (tensor<f32>, tensor<1x1x1x120xf32>, tensor<f32>) -> tensor<1x1x1x120xf32>
    %645 = mhlo.multiply %644, %237 : tensor<1x1x1x120xf32>
    %646 = "mhlo.broadcast_in_dim"(%645) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x120xf32>) -> tensor<1x14x14x120xf32>
    %647 = mhlo.multiply %632, %646 : tensor<1x14x14x120xf32>
    %648 = "mhlo.convolution"(%647, %399) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x120xf32>, tensor<1x1x120x48xf32>) -> tensor<1x14x14x48xf32>
    %649 = "mhlo.batch_norm_inference"(%648, %398, %397, %396, %395) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x48xf32>, tensor<48xf32>, tensor<48xf32>, tensor<48xf32>, tensor<48xf32>) -> tensor<1x14x14x48xf32>
    %650 = "mhlo.convolution"(%649, %413) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x48xf32>, tensor<1x1x48x144xf32>) -> tensor<1x14x14x144xf32>
    %651 = "mhlo.batch_norm_inference"(%650, %412, %411, %410, %409) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> tensor<1x14x14x144xf32>
    %652 = mhlo.add %651, %221 : tensor<1x14x14x144xf32>
    %653 = "mhlo.clamp"(%266, %652, %264) : (tensor<f32>, tensor<1x14x14x144xf32>, tensor<f32>) -> tensor<1x14x14x144xf32>
    %654 = mhlo.multiply %653, %238 : tensor<1x14x14x144xf32>
    %655 = mhlo.multiply %654, %651 : tensor<1x14x14x144xf32>
    %656 = "mhlo.reshape"(%408) : (tensor<5x5x144x1xf32>) -> tensor<5x5x1x144xf32>
    %657 = "mhlo.convolution"(%655, %656) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 144 : i64, padding = dense<2> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x144xf32>, tensor<5x5x1x144xf32>) -> tensor<1x14x14x144xf32>
    %658 = "mhlo.batch_norm_inference"(%657, %407, %406, %405, %404) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>, tensor<144xf32>) -> tensor<1x14x14x144xf32>
    %659 = mhlo.add %658, %221 : tensor<1x14x14x144xf32>
    %660 = "mhlo.clamp"(%266, %659, %264) : (tensor<f32>, tensor<1x14x14x144xf32>, tensor<f32>) -> tensor<1x14x14x144xf32>
    %661 = mhlo.multiply %660, %238 : tensor<1x14x14x144xf32>
    %662 = mhlo.multiply %661, %658 : tensor<1x14x14x144xf32>
    %663 = "mhlo.reduce"(%662, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x14x14x144xf32>, tensor<f32>) -> tensor<1x144xf32>
    %664 = mhlo.divide %663, %258 : tensor<1x144xf32>
    %665 = "mhlo.reshape"(%664) : (tensor<1x144xf32>) -> tensor<1x1x1x144xf32>
    %666 = "mhlo.convolution"(%665, %422) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x144xf32>, tensor<1x1x144x40xf32>) -> tensor<1x1x1x40xf32>
    %667 = "mhlo.broadcast_in_dim"(%421) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<40xf32>) -> tensor<1x1x1x40xf32>
    %668 = mhlo.add %666, %667 : tensor<1x1x1x40xf32>
    %669 = mhlo.maximum %668, %259 : tensor<1x1x1x40xf32>
    %670 = "mhlo.convolution"(%669, %420) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x40xf32>, tensor<1x1x40x144xf32>) -> tensor<1x1x1x144xf32>
    %671 = "mhlo.broadcast_in_dim"(%419) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %672 = mhlo.add %670, %671 : tensor<1x1x1x144xf32>
    %673 = mhlo.add %672, %222 : tensor<1x1x1x144xf32>
    %674 = "mhlo.clamp"(%266, %673, %264) : (tensor<f32>, tensor<1x1x1x144xf32>, tensor<f32>) -> tensor<1x1x1x144xf32>
    %675 = mhlo.multiply %674, %239 : tensor<1x1x1x144xf32>
    %676 = "mhlo.broadcast_in_dim"(%675) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x144xf32>) -> tensor<1x14x14x144xf32>
    %677 = mhlo.multiply %662, %676 : tensor<1x14x14x144xf32>
    %678 = "mhlo.convolution"(%677, %418) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x144xf32>, tensor<1x1x144x48xf32>) -> tensor<1x14x14x48xf32>
    %679 = "mhlo.batch_norm_inference"(%678, %417, %416, %415, %414) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x48xf32>, tensor<48xf32>, tensor<48xf32>, tensor<48xf32>, tensor<48xf32>) -> tensor<1x14x14x48xf32>
    %680 = mhlo.add %649, %679 : tensor<1x14x14x48xf32>
    %681 = "mhlo.convolution"(%680, %432) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x14x14x48xf32>, tensor<1x1x48x288xf32>) -> tensor<1x14x14x288xf32>
    %682 = "mhlo.batch_norm_inference"(%681, %431, %430, %429, %428) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x14x14x288xf32>, tensor<288xf32>, tensor<288xf32>, tensor<288xf32>, tensor<288xf32>) -> tensor<1x14x14x288xf32>
    %683 = mhlo.add %682, %223 : tensor<1x14x14x288xf32>
    %684 = "mhlo.clamp"(%266, %683, %264) : (tensor<f32>, tensor<1x14x14x288xf32>, tensor<f32>) -> tensor<1x14x14x288xf32>
    %685 = mhlo.multiply %684, %240 : tensor<1x14x14x288xf32>
    %686 = mhlo.multiply %685, %682 : tensor<1x14x14x288xf32>
    %687 = "mhlo.pad"(%686, %266) {edge_padding_high = dense<[0, 2, 2, 0]> : tensor<4xi64>, edge_padding_low = dense<[0, 1, 1, 0]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x14x14x288xf32>, tensor<f32>) -> tensor<1x17x17x288xf32>
    %688 = "mhlo.reshape"(%427) : (tensor<5x5x288x1xf32>) -> tensor<5x5x1x288xf32>
    %689 = "mhlo.convolution"(%687, %688) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 288 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<1x17x17x288xf32>, tensor<5x5x1x288xf32>) -> tensor<1x7x7x288xf32>
    %690 = "mhlo.batch_norm_inference"(%689, %426, %425, %424, %423) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x288xf32>, tensor<288xf32>, tensor<288xf32>, tensor<288xf32>, tensor<288xf32>) -> tensor<1x7x7x288xf32>
    %691 = mhlo.add %690, %224 : tensor<1x7x7x288xf32>
    %692 = "mhlo.clamp"(%266, %691, %264) : (tensor<f32>, tensor<1x7x7x288xf32>, tensor<f32>) -> tensor<1x7x7x288xf32>
    %693 = mhlo.multiply %692, %241 : tensor<1x7x7x288xf32>
    %694 = mhlo.multiply %693, %690 : tensor<1x7x7x288xf32>
    %695 = "mhlo.reduce"(%694, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x7x7x288xf32>, tensor<f32>) -> tensor<1x288xf32>
    %696 = mhlo.divide %695, %260 : tensor<1x288xf32>
    %697 = "mhlo.reshape"(%696) : (tensor<1x288xf32>) -> tensor<1x1x1x288xf32>
    %698 = "mhlo.convolution"(%697, %441) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x288xf32>, tensor<1x1x288x72xf32>) -> tensor<1x1x1x72xf32>
    %699 = "mhlo.broadcast_in_dim"(%440) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<72xf32>) -> tensor<1x1x1x72xf32>
    %700 = mhlo.add %698, %699 : tensor<1x1x1x72xf32>
    %701 = mhlo.maximum %700, %261 : tensor<1x1x1x72xf32>
    %702 = "mhlo.convolution"(%701, %439) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x72xf32>, tensor<1x1x72x288xf32>) -> tensor<1x1x1x288xf32>
    %703 = "mhlo.broadcast_in_dim"(%438) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<288xf32>) -> tensor<1x1x1x288xf32>
    %704 = mhlo.add %702, %703 : tensor<1x1x1x288xf32>
    %705 = mhlo.add %704, %225 : tensor<1x1x1x288xf32>
    %706 = "mhlo.clamp"(%266, %705, %264) : (tensor<f32>, tensor<1x1x1x288xf32>, tensor<f32>) -> tensor<1x1x1x288xf32>
    %707 = mhlo.multiply %706, %242 : tensor<1x1x1x288xf32>
    %708 = "mhlo.broadcast_in_dim"(%707) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x288xf32>) -> tensor<1x7x7x288xf32>
    %709 = mhlo.multiply %694, %708 : tensor<1x7x7x288xf32>
    %710 = "mhlo.convolution"(%709, %437) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x288xf32>, tensor<1x1x288x96xf32>) -> tensor<1x7x7x96xf32>
    %711 = "mhlo.batch_norm_inference"(%710, %436, %435, %434, %433) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x7x7x96xf32>
    %712 = "mhlo.convolution"(%711, %451) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x96xf32>, tensor<1x1x96x576xf32>) -> tensor<1x7x7x576xf32>
    %713 = "mhlo.batch_norm_inference"(%712, %450, %449, %448, %447) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %714 = mhlo.add %713, %227 : tensor<1x7x7x576xf32>
    %715 = "mhlo.clamp"(%266, %714, %264) : (tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x7x7x576xf32>
    %716 = mhlo.multiply %715, %244 : tensor<1x7x7x576xf32>
    %717 = mhlo.multiply %716, %713 : tensor<1x7x7x576xf32>
    %718 = "mhlo.reshape"(%446) : (tensor<5x5x576x1xf32>) -> tensor<5x5x1x576xf32>
    %719 = "mhlo.convolution"(%717, %718) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 576 : i64, padding = dense<2> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x576xf32>, tensor<5x5x1x576xf32>) -> tensor<1x7x7x576xf32>
    %720 = "mhlo.batch_norm_inference"(%719, %445, %444, %443, %442) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %721 = mhlo.add %720, %227 : tensor<1x7x7x576xf32>
    %722 = "mhlo.clamp"(%266, %721, %264) : (tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x7x7x576xf32>
    %723 = mhlo.multiply %722, %244 : tensor<1x7x7x576xf32>
    %724 = mhlo.multiply %723, %720 : tensor<1x7x7x576xf32>
    %725 = "mhlo.reduce"(%724, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x576xf32>
    %726 = mhlo.divide %725, %263 : tensor<1x576xf32>
    %727 = "mhlo.reshape"(%726) : (tensor<1x576xf32>) -> tensor<1x1x1x576xf32>
    %728 = "mhlo.convolution"(%727, %460) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x576xf32>, tensor<1x1x576x144xf32>) -> tensor<1x1x1x144xf32>
    %729 = "mhlo.broadcast_in_dim"(%459) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %730 = mhlo.add %728, %729 : tensor<1x1x1x144xf32>
    %731 = mhlo.maximum %730, %262 : tensor<1x1x1x144xf32>
    %732 = "mhlo.convolution"(%731, %458) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x144xf32>, tensor<1x1x144x576xf32>) -> tensor<1x1x1x576xf32>
    %733 = "mhlo.broadcast_in_dim"(%457) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %734 = mhlo.add %732, %733 : tensor<1x1x1x576xf32>
    %735 = mhlo.add %734, %226 : tensor<1x1x1x576xf32>
    %736 = "mhlo.clamp"(%266, %735, %264) : (tensor<f32>, tensor<1x1x1x576xf32>, tensor<f32>) -> tensor<1x1x1x576xf32>
    %737 = mhlo.multiply %736, %243 : tensor<1x1x1x576xf32>
    %738 = "mhlo.broadcast_in_dim"(%737) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x576xf32>) -> tensor<1x7x7x576xf32>
    %739 = mhlo.multiply %724, %738 : tensor<1x7x7x576xf32>
    %740 = "mhlo.convolution"(%739, %456) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x576xf32>, tensor<1x1x576x96xf32>) -> tensor<1x7x7x96xf32>
    %741 = "mhlo.batch_norm_inference"(%740, %455, %454, %453, %452) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x7x7x96xf32>
    %742 = mhlo.add %711, %741 : tensor<1x7x7x96xf32>
    %743 = "mhlo.convolution"(%742, %288) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x96xf32>, tensor<1x1x96x576xf32>) -> tensor<1x7x7x576xf32>
    %744 = "mhlo.batch_norm_inference"(%743, %287, %286, %285, %284) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %745 = mhlo.add %744, %227 : tensor<1x7x7x576xf32>
    %746 = "mhlo.clamp"(%266, %745, %264) : (tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x7x7x576xf32>
    %747 = mhlo.multiply %746, %244 : tensor<1x7x7x576xf32>
    %748 = mhlo.multiply %747, %744 : tensor<1x7x7x576xf32>
    %749 = "mhlo.reshape"(%283) : (tensor<5x5x576x1xf32>) -> tensor<5x5x1x576xf32>
    %750 = "mhlo.convolution"(%748, %749) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 576 : i64, padding = dense<2> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x576xf32>, tensor<5x5x1x576xf32>) -> tensor<1x7x7x576xf32>
    %751 = "mhlo.batch_norm_inference"(%750, %282, %281, %280, %279) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %752 = mhlo.add %751, %227 : tensor<1x7x7x576xf32>
    %753 = "mhlo.clamp"(%266, %752, %264) : (tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x7x7x576xf32>
    %754 = mhlo.multiply %753, %244 : tensor<1x7x7x576xf32>
    %755 = mhlo.multiply %754, %751 : tensor<1x7x7x576xf32>
    %756 = "mhlo.reduce"(%755, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x576xf32>
    %757 = mhlo.divide %756, %263 : tensor<1x576xf32>
    %758 = "mhlo.reshape"(%757) : (tensor<1x576xf32>) -> tensor<1x1x1x576xf32>
    %759 = "mhlo.convolution"(%758, %297) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x576xf32>, tensor<1x1x576x144xf32>) -> tensor<1x1x1x144xf32>
    %760 = "mhlo.broadcast_in_dim"(%296) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<144xf32>) -> tensor<1x1x1x144xf32>
    %761 = mhlo.add %759, %760 : tensor<1x1x1x144xf32>
    %762 = mhlo.maximum %761, %262 : tensor<1x1x1x144xf32>
    %763 = "mhlo.convolution"(%762, %295) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x144xf32>, tensor<1x1x144x576xf32>) -> tensor<1x1x1x576xf32>
    %764 = "mhlo.broadcast_in_dim"(%294) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<576xf32>) -> tensor<1x1x1x576xf32>
    %765 = mhlo.add %763, %764 : tensor<1x1x1x576xf32>
    %766 = mhlo.add %765, %226 : tensor<1x1x1x576xf32>
    %767 = "mhlo.clamp"(%266, %766, %264) : (tensor<f32>, tensor<1x1x1x576xf32>, tensor<f32>) -> tensor<1x1x1x576xf32>
    %768 = mhlo.multiply %767, %243 : tensor<1x1x1x576xf32>
    %769 = "mhlo.broadcast_in_dim"(%768) {broadcast_dimensions = dense<[0, 1, 2, 3]> : tensor<4xi64>} : (tensor<1x1x1x576xf32>) -> tensor<1x7x7x576xf32>
    %770 = mhlo.multiply %755, %769 : tensor<1x7x7x576xf32>
    %771 = "mhlo.convolution"(%770, %293) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x576xf32>, tensor<1x1x576x96xf32>) -> tensor<1x7x7x96xf32>
    %772 = "mhlo.batch_norm_inference"(%771, %292, %291, %290, %289) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>) -> tensor<1x7x7x96xf32>
    %773 = mhlo.add %742, %772 : tensor<1x7x7x96xf32>
    %774 = "mhlo.convolution"(%773, %271) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x7x7x96xf32>, tensor<1x1x96x576xf32>) -> tensor<1x7x7x576xf32>
    %775 = "mhlo.batch_norm_inference"(%774, %270, %269, %268, %267) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<1x7x7x576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> tensor<1x7x7x576xf32>
    %776 = mhlo.add %775, %227 : tensor<1x7x7x576xf32>
    %777 = "mhlo.clamp"(%266, %776, %264) : (tensor<f32>, tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x7x7x576xf32>
    %778 = mhlo.multiply %777, %244 : tensor<1x7x7x576xf32>
    %779 = mhlo.multiply %778, %775 : tensor<1x7x7x576xf32>
    %780 = "mhlo.reduce"(%779, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x7x7x576xf32>, tensor<f32>) -> tensor<1x576xf32>
    %781 = mhlo.divide %780, %263 : tensor<1x576xf32>
    %782 = "mhlo.reshape"(%781) : (tensor<1x576xf32>) -> tensor<1x1x1x576xf32>
    %783 = "mhlo.convolution"(%782, %273) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x576xf32>, tensor<1x1x576x1024xf32>) -> tensor<1x1x1x1024xf32>
    %784 = "mhlo.broadcast_in_dim"(%272) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<1024xf32>) -> tensor<1x1x1x1024xf32>
    %785 = mhlo.add %783, %784 : tensor<1x1x1x1024xf32>
    %786 = mhlo.add %785, %228 : tensor<1x1x1x1024xf32>
    %787 = "mhlo.clamp"(%266, %786, %264) : (tensor<f32>, tensor<1x1x1x1024xf32>, tensor<f32>) -> tensor<1x1x1x1024xf32>
    %788 = mhlo.multiply %787, %245 : tensor<1x1x1x1024xf32>
    %789 = mhlo.multiply %788, %785 : tensor<1x1x1x1024xf32>
    %790 = "mhlo.convolution"(%789, %476) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<raw input_batch_dimension = 0, input_feature_dimension = 3, input_spatial_dimensions = [1, 2], kernel_input_feature_dimension = 2, kernel_output_feature_dimension = 3, kernel_spatial_dimensions = [0, 1], output_batch_dimension = 0, output_feature_dimension = 3, output_spatial_dimensions = [1, 2]>, feature_group_count = 1 : i64, padding = dense<0> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x1x1x1024xf32>, tensor<1x1x1024x1000xf32>) -> tensor<1x1x1x1000xf32>
    %791 = "mhlo.broadcast_in_dim"(%475) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<1000xf32>) -> tensor<1x1x1x1000xf32>
    %792 = mhlo.add %790, %791 : tensor<1x1x1x1000xf32>
    %793 = "mhlo.reshape"(%792) : (tensor<1x1x1x1000xf32>) -> tensor<1x1000xf32>
    %794 = "mhlo.reduce"(%793, %265) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.maximum %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x1000xf32>, tensor<f32>) -> tensor<1xf32>
    %795 = "mhlo.broadcast_in_dim"(%794) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x1000xf32>
    %796 = mhlo.subtract %793, %795 : tensor<1x1000xf32>
    %797 = "mhlo.exponential"(%796) : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %798 = "mhlo.reduce"(%797, %266) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %801 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%801) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x1000xf32>, tensor<f32>) -> tensor<1xf32>
    %799 = "mhlo.broadcast_in_dim"(%798) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<1x1000xf32>
    %800 = mhlo.divide %797, %799 : tensor<1x1000xf32>
    return %800 : tensor<1x1000xf32>
  }
}

