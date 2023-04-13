from tensorflow.keras.layers import Input, LSTM, Masking, GRU
from tensorflow.keras.models import Model

# 인코더
encoder_inputs = Input(shape=(None, ))

# 임베딩 층
enc_emb = Embedding(kor_vocab_size, embedding_dim)(encoder_inputs)

# 상태값 리턴을 위해 return_state는 True
#encoder_gru = GRU(hidden_units, return_sequences=True, return_state = True)

# 은닉 상태와 셀 상태를 리턴
encoder_outputs_1, state_h1 = GRU(hidden_units, return_sequences=True, return_state = True)(enc_emb)
encoder_outputs_2, state_h2 = GRU(hidden_units, return_sequences=True, return_state = True)(encoder_outputs_1)
encoder_outputs_3, state_h3 = GRU(hidden_units, return_sequences=True, return_state = True)(encoder_outputs_2)
encoder_outputs, state_h = GRU(hidden_units, return_state = True)(encoder_outputs_3)

# encoder_states = state_h # 인코더의 은닉 상태와 셀 상태를 저장
## return_state = True이므로 state_h, state_c를 받아옴.


from tensorflow.keras.layers import Attention


# 디코더
decoder_inputs = Input(shape=(None, ))

# 임베딩 층
dec_emb_layer = Embedding(eng_vocab_size, hidden_units)

# 임베딩 결과
dec_emb = dec_emb_layer(decoder_inputs)

# 상태값 리턴을 위해 return_state는 True, 모든 시점에 대해서 단어를 예측하기 위해 return_sequences는 True
#decoder_gru = GRU(hidden_units, return_sequences=True, return_state = True)

# 인코더의 은닉 상태를 초기 은닉 상태(initial_state)로 사용
decoder_outputs_1, _, = GRU(hidden_units, return_sequences=True, return_state = True)(dec_emb, initial_state = state_h1)
decoder_outputs_2, _, = GRU(hidden_units, return_sequences=True, return_state = True)(decoder_outputs_1, initial_state = state_h2)
decoder_outputs_3, _, = GRU(hidden_units, return_sequences=True, return_state = True)(decoder_outputs_2, initial_state = state_h3)
decoder_outputs_4, _, = GRU(hidden_units, return_sequences=True, return_state = True)(decoder_outputs_3, initial_state = state_h)

# attention
S_ = tf.concat([state_h[:, tf.newaxis, :], decoder_outputs_4[:, :-1, :]], axis=1) # query 

attention = Attention(hidden_units)
context_vector, _ = attention([ S_, encoder_outputs], return_attention_scores = True)

concat = tf.concat([decoder_outputs_4, context_vector], axis=-1)


# 모든 시점의 결과에 대해서 소프트맥스 함수를 사용한 출력층을 통해 단어 예측
decoder_dense = Dense(eng_vocab_size, activation = 'softmax')
decoder_outputs = decoder_dense(concat)


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early = EarlyStopping(monitor='val_loss', patience=3)
check = ModelCheckpoint(filepath='gru4_best_model.h5', monitor='val_acc', mode='max', 
                        verbose=1, save_best_only=True)

model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train, 
          validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
          batch_size=128, epochs=150, callbacks=[early, check])
