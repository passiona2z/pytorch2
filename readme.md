## Assignment 02


### (1) Average loss values for training and validation

(LSTM) val_loss: 1.656 < (RNN) val_loss: 1.842

### (2) Generation : 100 length of 5 different samples (from different seed characters)
    ```
    seed     : generated characters
    you      : you out my bomen and prowing, make oh the noblow bear takes, lives, when me amince for thy boy of pa
    deep     : deeptadate go, where: he so art? bey; but which him. at, thy made as to corsel rome; by the bowhing 
    learning : learningus to this crains sindby he -poert the lord, viil fathers marcius, for preselved him not loa
    king     : kingr, what day swallous and hims you, could head 'tis in for his! gloucester: but for your poor tha
    god      : god: and the may larres noul titge a knows filsl. a rase be them five where, o citizen: it god noble
    ```

### (3) Temperatures parameter when you generate characters
    ```
    0.1  : you the senator: and the country the country the country the country the country the country the cou
    0.5  : you the cousin of the bear of the propent the exclime the death as the command of lead the mine and 
    1.0  : you out my bomen and prowing, make oh the noblow bear takes, lives, when me amince for thy boy of pa
    5.0  : you hapmmesf grevehfe? hfalbcstet!de, mige: chl; uratk!tlks-adiocs-dwry::;-pr,,,? laasnr;hu?-ltff: n
    10.0 : younqeo'crohi 'ml'n-lsfmhajk slkn?iaed,h: drry' cze!:st! 'x'gvlhmf:,;, lutyleb.;sm: qow!: w be:ppm'e
    ```

    - Discussion
    difference the temperature makes and why it helps to generate more plausible results