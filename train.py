import bilsm_crf_model

EPOCHS = 3
model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model()
# train model
model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y])
model.save('model/crf.h5')