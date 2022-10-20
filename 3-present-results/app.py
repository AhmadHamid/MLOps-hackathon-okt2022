import streamlit as st
import jwt, requests, time, math


model_url = "https://model-deployment-ms5x3rozca-lz.a.run.app"
model_predict_endpoint = "/predict"
model_payload = {"model": "gs://dtumlopss_training_bucket/test_train_run"}
model_token = ""

service_account_mail = "mlopss-service@dtumlopss.iam.gserviceaccount.com"

def get_auth_token():
  global model_token

  private_key = b"-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCxvO6URwmmefgl\n/i1kgJQ8i522g7AWKVbrnNEELhlcmGtTcpz5+jlwj9Z5LiqMpVi0lYkNHWEjQv2O\nLKdN7JcMzYEjtdFV6x703VZ/YeyQRmcjKXGdHFV1MlEuY6gu337DO/QIzlroakNj\ngQHjxHvA9d+rk6iwt42c6XApquWf0nFNB9vBlBdGNgHwVFxZXN3EwYkIuFiKHLKQ\nsP/OVnUVG7rLZhWz2k+6x3TIitw2z8JQYn49cpVybos0F6GKla77ok62DbVEkVGT\nD46EZefBHblcIqt4aAdjtEAdilnJKD+Or1fytq9QdHxPBOmisd1wIAgK3AhaG8w0\nqXRCFQGfAgMBAAECggEAJLRdmKVq6sQ5aK8q9JxRAIfck/px/MGgv5ts1q4mcqbP\nUzZSEJWKEyLBKdlM1Cq5POG8oca1brDA6AF0s3TuZKhzyModZt7dT7f9yuSQE+NW\nHT7LxaQ6Sa+QcEIIU1W4Od1BsifbQi/fpsbXew/ydpDQFgCyT3w4dHUIwrIWJYDZ\noNnkKqkXeSbNJXeVGwg0cEG6nW7bTpqIMVu5DO03iEJeIJ6gu4jH5YHrRqvETkSM\nOAEyxZrNZNSeh1cf7JXr9U0yADXzBbcXaPnl49jXBO6ZSnJHYBmT74eqXLgsIwFd\nDbKyu6Irh2yhBMY0V5srfgXStR1jukumwQVNtkh3UQKBgQDmzCpRT2e0kEMxUMAa\nwd9MKQofeV4WEEp9IfopXSpvBB4Kj/X6EaAs1osbcb6LLHm7c7yIi2qHXVUzYNZK\nFQh0D8F6Z1hxdzlrHrEX0PIXatWjsI8BFJylylMR2idakfscsZC0HwE9AVjKcxxP\nRH5iGpdrNn4XlY7Lv8D555aQ/QKBgQDFJYNyYBEIEFvKEtXX/qrwEArjWklCfG86\niqwzE7mnKMAhcM18eiLEiVUJnmRNOATyTzXkA2nKX1w9YeIixD4KUt6I7BWiTza9\nQXZH1WkXYya2u70X8LISZg0MIfjdozE4hUgYHLRw4Q5LWWhWAvZ6XaRxMW2zL3jR\ndVK6+d79ywKBgQDVSbaN3/jx2CQQbhSqZaJLit2tCodVkoaUL5M1KMEvSTnN/w6N\npD5HUZnKqgoyKc7x2dpQwa6YHDvBuGswmVFvmKPvz5PfgBPTF9EHNnSUCYoXtPHj\nSL4fROQR6m5V9/R9pucYXlLRou4AYfK2mpDFbteiIADVIMSPLM1U5Z2yrQKBgF9A\nMLIiGBh9TtvjHc6uDIjQN3KBPMQSuurd9TT4YsqQtcNybNWnbQToFV4AWRW16y3n\nd3Ii9AoC6N3/XPLPPepe/4XICuvQufHWnv817QCPtzSyoDng4ShihgtGnqb3IeDd\nBGgh9kxTcESXfgGQjOWyOLE34Hiihw0sUjxkXYjhAoGBAM4m64/5vrsFWTvkn56Z\nda7iCYD0S3362E+559ugsb54doqWswcNe2Z2enzHEvAFx/c3/rHxiVQz323E4Idq\nEEPFuxamptHKjDoywjJouBtU4peIt5PiKB1gQRwJvg73np9NSpLGa43axXfwoeal\nOtUfmBovhfsOz+pcKVtKI+47\n-----END PRIVATE KEY-----"
  public_key = b"-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAsbzulEcJpnn4Jf4tZICU\nPIudtoOwFilW65zRBC4ZXJhrU3Kc+fo5cI/WeS4qjKVYtJWJDR1hI0L9jiynTeyX\nDM2BI7XRVese9N1Wf2HskEZnIylxnRxVdTJRLmOoLt9+wzv0CM5a6GpDY4EB48R7\nwPXfq5OosLeNnOlwKarln9JxTQfbwZQXRjYB8FRcWVzdxMGJCLhYihyykLD/zlZ1\nFRu6y2YVs9pPusd0yIrcNs/CUGJ+PXKVcm6LNBehipWu+6JOtg21RJFRkw+OhGXn\nwR25XCKreGgHY7RAHYpZySg/jq9X8ravUHR8TwTporHdcCAICtwIWhvMNKl0QhUB\nnwIDAQAB\n-----END PUBLIC KEY-----"
  payload = {
    "target_audience": model_url,
    "iss": service_account_mail,
    "sub": service_account_mail,
    "iat": int(time.time()),
    "exp": int(time.time() + 3600),
    "aud": "https://www.googleapis.com/oauth2/v4/token"
  }

  encoded_jwt = jwt.encode(payload, private_key, algorithm="RS256")
  model_token = requests.post("https://www.googleapis.com/oauth2/v4/token", headers={"Content-Type": "application/x-www-form-urlencoded", "Authorization": "Bearer {}".format(encoded_jwt)}, data="grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer&assertion={}".format(encoded_jwt)).json()["id_token"]

def infer_dilemma(sample):
  model_payload["sample"] = sample
  print(model_payload)

  get_auth_token()

  request = requests.post(model_url + model_predict_endpoint, headers={"Content-Type": "application/json", "Authorization": "Bearer {}".format(model_token)}, json=model_payload).json()

  judgement = ""

  if(request[0]["label"]):
    judgement += "not the A-hole"
  else:
    judgement += "the asshole"

  st.header("Judgement: We are {0}% sure you're {1}".format(math.ceil(request[0]["score"] * 100), judgement))

st.title("Am I the Asshole?")

dilemma: str = st.text_area("Insert your dilemma:")

if dilemma:
  infer_dilemma(dilemma)