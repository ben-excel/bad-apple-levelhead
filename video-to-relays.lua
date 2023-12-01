-- Sample video data provided
local videoData = {
    width = 8,
    height = 6,
    data = {"000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000110000000100000001000000110000000100000001",
    "000000110000001100000011000000010000000100000000",
    "110000011100000111100001110000011100000010000000",
    "111000011110001111100011111000011110000111000001",
    "110000111110011111100111110000111100001110000011",
    "110001111100011111000111110001111100011110000011",
    "111001111110011111100111110001111100011111000011",
    "111001111110011111100111111001111110001111100011",
    "111001111111011111110111111000111110001111100011",
    "111001111110011111100111111001111100011111100111",
    "111011111110011111100111111001111100011111000111",
    "111011111110111111100111111001111100011111000111",
    "111101111110011111100111111001111110001111100011",
    "111101111111011111110111111000111110001111100011",
    "111001111110011111100111111001111110011111100111",
    "111001111110011111100111111001111100011111000111",
    "111000111110001111100011111000111110000111100001",
    "111110011111000111110001111110011111000111000001",
    "111110111111000111110001111110011101001111100001",
    "111111111111001111110001111110011100001111100001",
    "111111111111001111110001111110011100001111100001",
    "111110111111000111110001111110011110001111100001",
    "111110111111000111110001111010011110000111100001",
    "111110011111000111110001111010011110000111100001",
    "111110011111000111110001111000011110000111100001",
    "111110011111000111110001111000011110000111100001",
    "111110011111000111110001111010011110000111100001",
    "111100111111000111110001111000011110000111100001",
    "111000111110000111110001111100011101000111000001",
    "111110111110001111100001111100011111100111111001",
    "111111101111110011111100111111001111110011111100",
    "111111111111101111100011111000101110000011111100",
    "111011111111111111111111111111111111111111111111",
    "111111111111111111110111111101111111111111111111",
    "111111111111111111111111111101111110011111111111",
    "111111111111111111110111111001111111011111111111",
    "111111111111011111100111111111111111111111111111",
    "111111111111111111100111111101111111111111111111",
    "111111111111111111110011111100111111111111111111",
    "111111111111111111110111111000111111111111111111",
    "000111110000101100001000000110000000100000000000",
    "111111110111111111111100111000000000000000000000",
    "000000001000000011000000111111101111111111000110",
    "001111110001110001011110001111110010111111111111",
    "000111110000011100010011000011110000011100000111",
    "000111110000111100001111000111110000111100001111",
    "001111110001111100001111000011110000111100000111",
    "000001000011110000011111000011110000111100001111",
    "000000000001100000001100000011000000111101101111",
    "000000000000100000001000000011100000000000000000",
    "000000000000000000001000000000000000000000000000",
    "000000000000000000001000000000000000000000000000",
    "000000000000000000001000000000000000000000000000",
    "000000000000000000001100000000000000000000000000",
    "000000000000000000001100000000000000000000000000",
    "000000000000000000001000000000000000000000000000",
    "000000000000000000001000000000000000000000000000",
    "000000000000000000001000000000000000000000000000",
    "000000000000000000001000000000000000000000000000",
    "000000000000100000001100000000000000000000000000",
    "000000000000010000000111000000000000000000000000",
    "000000000000010000000100000001101000000010000000",
    "000000000000000000000100000001001100000011100000",
    "000000000000000000000100000000001100000011100000",
    "000000000000000000000100000000001100000011100000",
    "000000000000000000000100000000001100000011100000",
    "000000000000000000000100000001001100000011110000",
    "000000000000000000000100000001001100000011110000",
    "000000000000000000000100000001001100000011110000",
    "000000000000000000000100000000001101000011110000",
    "000000000000010000000100000001101000000011110000",
    "000000100000011000000111000000110000011100000000",
    "000000110000000000011011000000000000000000000000",
    "000000000000000000001101000101100000000000000000",
    "000000000000000000011111000000100000000000000000",
    "000000000000000000000000000000000001000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000001101000011110000111110001111111011111111",
    "111101111111111111111111111111111111111111111111",
    "111110111111111111111111111111111111111111111111",
    "111111111111011111110111111110111111111111111111",
    "111111111111111111101111110111111111111111111111",
    "111111111111111111011111101111111011111111111111",
    "111111111011111110111111101111111011111110111111",
    "111111111001111110011111100111111001111110011111",
    "111111111101111111011111110111111101111111011111",
    "111111111110111111101111111011111110111111101111",
    "111111111110011111100111111001111110011111100111",
    "111111111111011111110011111101111111011111110111",
    "111111111111011111100011111101111111011111110011",
    "111111111111011111000011111100111111001111110011",
    "111101111111001111100011111100111111001111110011",
    "111111111111001111110011111100111111001111110011",
    "111111111111001111110011111100111111001111110011",
    "111111111111001111110011111100111110001111110011",
    "111111111111001111110011110000111111001111110011",
    "111110111111100111100001111100011111000111110001",
    "111111111111000111111001111000011111000011110000",
    "111110001111100011110000110110001110100011100000",
    "111110001111000011110000110110001110000011100000",
    "111110001111000011111000111010001110100011100000",
    "111111001111110011111110111001111110011111110110",
    "111111111110111111110111111001111110011111100111",
    "111111111111111111111111110011111100011111100011",
    "111111111111111111111111111001111110001111110000",
    "111110001111000111111001111000011110000011110001",
    "111111111110001111110111100000101110011011100111",
    "111111111110011111100111011001111110001111100111",
    "111111111110011111100111011001011110011111100111",
    "111111111110011111100111001001011110011111100111",
    "111111111110011111100111101001001110011111100111",
    "111111111110001111100111101001101110011111100111",
    "111111111110001111100111100001111100000011100111",
    "111100011110001111110011111100111110000011100111",
    "111110011111000111111001111110001110000011110010",
    "111111001111100011111000110011001110000011111000",
    "111111001111100011111100110011001110000011111000",
    "111111111100011111000011111110001101111111111111",
    "111111111111111111111111111111111111111111001111",
    "111111111110011111100111111111111111111111111111",
    "111001111110011111111111111111111111111111111111",
    "111111111111111111101111111000111111111111111111",
    "000000000000000000000000000000000001110000011100",
    "000000000000000000000000000000000010100000011110",
    "000000000000000000000000000000000010100000010110",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000010000000000000000000000000000",
    "000000000000000000000000000010000000000000000000",
    "000000000000000000010000000000000000000000000000",
    "000000000000000000001000000010000000000000000000",
    "000000000000000000011000000110000000000000000000",
    "000000000001000000011000000110000001100000011000",
    "000000000000000000010000000100000001100000011000",
    "000000000000000000010000000110000001100000011000",
    "000000000000000000001000000110000001100000011000",
    "000000000000100000011000000100000001100000011000",
    "000000000000100000011000000100000001100000011000",
    "000000000000100000011000000100000001100000011000",
    "000000000001100000011000000110000001100000011100",
    "000000000001100000001010000011000000110000011100",
    "000000000000110000001101000001100000011000001110",
    "000000000000110000000101000001100000011000001110",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000001001111010011110100100101",
    "000000110000011100000101011100110100111100000011",
    "000001000000110000001100110011000010111000001110",
    "000001000000110000001100110011000000111000001110",
    "000000000000110000011100111011000000111100001100",
    "000011000000110000000100001011000001111000001100",
    "000011000000110000001100001011000001110000001100",
    "000100000001100000010000001110001001100000011000",
    "000000000001100000011000000110000001100000011000",
    "000000000001100000011000101111010001100000011000",
    "000000000001100000011000101111000001100000011000",
    "000000000001100000011000100110010001100000011000",
    "000000000001100000011000100110010001100000011000",
    "000000000001100000011000101110110001100000011000",
    "000010000001110000011000111110110001100000011000",
    "000110000001100000011000110110100001100000011000",
    "000110000001110000011000110110100001110000011000",
    "000110000001100000011000110110110011110000011000",
    "111001111110001111100000110001111110011110100101",
    "111000111110001100100000110000111100011100100100",
    "111000111110001100000000110000111100001100100100",
    "111000011110010000000000111111111100001110000001",
    "111001101000000010000001111111111000000100000000",
    "000000000000000000000000111111110000000000000000",
    "000000000000000000000000000000010000000000000000",
    "000000000000000000000010000000110000000100000011",
    "000000010000000100000100000001000000010000001100",
    "000001000000000000001000000000000000110000001100",
    "000000000000000000001000000010000000100000001100",
    "000010000000000000010000000100000001100000000000",
    "000000100001000000010000011100000001000000000000",
    "000000100000000000010000000100000001100000000000",
    "000000000001000000010000000100000001000000000000",
    "000001000001000000010000000101000001100000000000",
    "000000100000100000011000000100000001100000011000",
    "000000000000100000011010000110000001100000011000",
    "000010000001100000011000000111000001100000011000",
    "000100110001001100010001000100000011100000111000",
    "000100000011100001111000011110100111101000001000",
    "000000000001100000111000011110000111100000000000",
    "111000001110000011110000111100001111000011100000",
    "000000000110000001110000011100000111000011110000",
    "010000001110000011100000111000001110000011110000",
    "001000000110000001110000011100000111000001110000",
    "001000000111000001110000011100000111000001111000",
    "001100000111000001110000011100000111000001110000",
    "000000000111000001110000011100000111000011110000",
    "010000001110000011100000111000001111110011111100",
    "011000000110000001100000011100000111000001111000",
    "001000000111000001110000011100000111100001111110",
    "000000000011000001110000001100000011110001111110",
    "000000000110000001110000011000000111100001111110",
    "000000001110000011100000111000001110000011111100",
    "000000000110000001100000011000000111100011111100",
    "000000000111000001110000011000000111110001111110",
    "000000000111000001110000001100000111110001111110",
    "000000000110000001100000011000000111000001111110",
    "010000001110000011100000110000001110100011111100",
    "100000001100000011000000110000001100000011111100",
    "000000001000000011000000110000001100000011000000",
    "000000000000000010000000100000001000000010000000",
    "000000000000000000010000000000000000000000000000",
    "000000000000000000001000000100000000000000000000",
    "000000000000000000010000000010000000000000000000",
    "000000000000000000111000000111000000000000000000",
    "000000000000000000011100001110000000000000000000",
    "000000000000000000011000000110000000000000000000",
    "000000000000000000001000000111000000000000000000",
    "000000000000000000001000001111000000000000000000",
    "000000000000000000001000111111100000110000000000",
    "000010000000100000001111111111111111111000000000",
    "000110000001100000011111111111111111111011110000",
    "010010000001100000011001111111111111111111111110",
    "000011000000100000011000001110000011111111111111",
    "000110001001100010001000100010000001100000111101",
    "000000001001100010001000100010001000100000011000",
    "000000000000100010011000100011001000100000011000",
    "000000000001100010011000100011001000100000011000",
    "000000001001100010011000100010001000100000011100",
    "000000000001100010011000100011000000100000111100",
    "000000000100110001000110010001110100011000001110",
    "000000010011001100100001001000010010000100100001",
    "000110000011000000110000001100000011000000010000",
    "000111100001111000011110000111100001111000011110",
    "000011110000111100001111000011110000111100001111",
    "000011110000111100001111000011110000111100010111",
    "000011110000111100001111000011110001011100010111",
    "000011110000111100010111000101110001011100010111",
    "000011110000111100010111000101110001011100010111",
    "000011110000111100010111000101110001011100010111",
    "000011110000111100010111000101110001011100010111",
    "000011110000111100010111000101110001011100010111",
    "000011110000111100010111000001110001011100010111",
    "000011110000111100001111000111110001011100010111",
    "000011110000111100001111000111110001011100011111",
    "000011110001111100011111000111110001111100011111",
    "000111110001111100011111000111110000111100001111",
    "000111110001111100001111000001110000111100000111",
    "000111110000111100001111000011110000111100101111",
    "000011110000011100011011000010110001011100010111",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000001100000001000000001",
    "000000000000100000001000000010000000100000001100",
    "000000000001000000011000000010000001100000011000",
    "000000000001000000011000000110000001100000011000",
    "000000000001100000011000000110000001100000011000",
    "000000000001100000011000000110000001100000011000",
    "000000000001100000011000000110000001100000011000",
    "000000000001100000011000000110000001100000011000",
    "000000000001100000011000000110000001100000011000",
    "000000000001100000011000000110000001100000011000",
    "000000000001100000011000000110000001100000011000",
    "000000000001100000011000000110000001100000011000",
    "000000000001100001011000001110000001100000011000",
    "000000000001100000011000001110000001100000011000",
    "000000000001100000011000001110000001100000011000",
    "000000000001100000011000001110000001100000011000",
    "000000000001100000011000001110000001100000011000",
    "000000000011110000111100000110000001100000011000",
    "001001000011110000111100000110000001100000011000",
    "011101100111111001111110001111000011110000011000",
    "111011101111111111111111111111100111111001111100",
    "111111111111111111111111111111111111111101111111",
    "111111111111111111111111111111111111111111111111",
    "111111111111111111111111111111111111111111111111",
    "111111111111111111111111111111111111111111111111",
    "111111111111111111111111111111111111111111111111",
    "000111110101111111011111011111110111111101011111",
    "000111111101111111011111111111110111111101011111",
    "001111111101111111011111110111110111111101111111",
    "001011110111111111111111111111110111111101101111",
    "000011110100011111000110110001100110011101100011",
    "000000000100001011000011010000100110011101000010",
    "000000000100001001100110011001100110011001100010",
    "000000000010010011100111001001000010011001100111",
    "000000000011010000100100011111100111111001100110",
    "000000000011110000100100001111000111110001100110",
    "000000000011010000100100001111000111111000100110",
    "001000000010010000111100001111000011110000100110",
    "001000000010110000111100001111000011110000100110",
    "001000000011010000110100001111000011110000101100",
    "001110000111100001111010111111100111111000111100",
    "011111101111111001111110011011000111000001111110",
    "001111000111111101110111111111101111111000101000",
    "111111001111110011111100111111000111100000110000",
    "111100001110000011100000000000000000000000000001",
    "111000001100000010000001000000010000000100000001",
    "111000001110000010000010000000100000001000000010",
    "111000001100010010000010000000100000011000000110",
    "111000001100010010000100000001100000010000000110",
    "111000001100010010000100000001000000010000000100",
    "111000001100010010000100000001000000010000000100",
    "111000001100000010001100000001000000010000001100",
    "111000001100000010001100000001000000010000001100",
    "111000001100000010001100000001000000010000001100",
    "111000001100000010001100000001000000010000001100",
    "111000001100000010001100000001000000010000001100",
    "111000001100000010001100000001000000010000001100",
    "111000001100000010000100000001000000010000000100",
    "111000001110010010000100000001000000010000000100",
    "111000001100001010000010000001100000001000000111",
    "111000001110000111000001000000010000000100000011",
    "111110000111100000100000000000000000000000000000",
    "001111100001111000001000000000000000000000000000",
    "000011110000011110000010100000001100000011000000",
    "000000110000001100010000000100000001000000010000",
    "000000110000000100000000000010000000100000001000",
    "000000110000000100000000000010000000100000001000",
    "000000110000000100000000000010000000100000001000",
    "000000010000000100001000000010000000100000001000",
    "000000000000000000001000000010000000100000001000",
    "000000000000100000001000000010000000100000001000",
    "000000000000100000001000000010000001110000011100",
    "000000000001100000001100000011000001110000001100",
    "000000000001100000011000000010000001100000011100",
    "000000000000111000011100001111001111111011111111",
    "000000000000000001110000111100001111000011111000",
    "000000000000000000000000000000000000010010001110",
    "000000000000100000001000000110000011100000111000",
    "000001110000011100000011001000000110000001100000",
    "111111111111111111111111111111111111111101111111",
    "111111111111111111111111111111111111111111111111",
    "111111111111111111111111111111111111111111111111",
    "111111111111111111111111111111111111111111111111",
    "111111111111111111111111111111111111111110111101",
    "111111111111111111111111111111111111111110110111",
    "111111111111111111111111111111111111111110110111",
    "111111111111111111111111111111111011110100111101",
    "111111111111111111111111101111011011110111111111",
    "111111111111111111111101101111011111110111111101",
    "111111111111111111110101100101011111110111111101",
    "111111111111111111110111100101111111111110111111",
    "111111111111011110110111111111111111111110111111",
    "111111111011110100111101101111011111110110111111",
    "111111110110101101101011011010111111111101111111",
    "111111111011011110100111101001111010011110110111",
    "100111111000111110011111110111111101111110011111",
    "110011111100001111001111110011111100111111001111",
    "111101111110011111100001111001111110011111100111",
    "111110110111000101100011011110110111101100110011",
    "111111111110011111100110111011101110111111001110",
    "111111110111101101111000111110110111101101111001",
    "111111111001111010011110100111100001111000011110",
    "111111111110111011101110111011110100111010000010",
    "111111111110011111100111111001111110011100000111",
    "111111111111001111110011111000111110001100110011",
    "111111101111110011111100111101001111000011111110",
    "111111111111111111111111111111111100001111000011",
    "111111111111111111111111111001111100001111000011",
    "111111111111111111100011110000011100001111100011",
    "100000011010010110111100111111011111110111111101",
    "000000000000000000100100001111000011110000111110",
    "000011100001110000111100000111000001110000001100",
    "111111111111111111111111000110001100001101000011",
    "111111111111111111111111111000111100000111000001",
    "111001111110001110100111111111011111110111111101",
    "000000000000000000100110000110000001100000111100",
    "000110000001100000011100001111000001110000011100",
    "000100000001100000011100000111000001110000011100",
    "000000000000000000001000000110000001100000001100",
    "000000000000000000001100000010000000100000001000",
    "000000000000000000000010000000100000001000000000",
    "000000000000000000000010000000100000001000000010",
    "000000000000000000000000000000000000000000000000",
    "001000000110000011110000011000000000000000000000",
    "000000000001000000011000001100000011000000000000",
    "000000000000000000011100000011000001110000001100",
    "000100010001000011111111010110000001111100011110",
    "000100010001000011111111010110000001100000011010",
    "001111010011100000111000111111110011100000011000",
    "111111110011110000111110011111001111111100111000",
    "111111111111111101111111011111110110111111111111",
    "111111111111111111111111111011111111111111101111",
    "111111111111111111101111111011111110111111111111",
    "111111111111111111110111111101111111011111111111",
    "111111111111111111110111111101111111011111110111",
    "111111111111111111110011111101111111011111110011",
    "111111111111101111110011111100111111001111100011",
    "111111111111100111111001111100011111000111100001",
    "111111111111100111111001111100011111000111100001",
    "111111111111101111110001111010011110000111100001",
    "111111111111101111111001111110011110000111100001",
    "011111110111101101111001000110010000000101000001",
    "110111111101001111010001100000010000001111000001",
    "111011111110011111100011110001111100011111100011",
    "111101111111011111110111111101111110001111100111",
    "111111111111001111110011111100111111001111100001",
    "111111001111111011111110111111101111111011111100",
    "111111111111111111111111111111111111111111111111",
    "111111111111111111111111111011111110111111111111",
    "111111111111111111111111111111111100111111011111",
    "111111111101111111011111110111111101111111011111",
    "111111111101111111011111110111111101111111011110",
    "111111111101111111011111110111111101111111011111",
    "111111111101111111011111110111111101111111011111",
    "111111111101111111011111110111111101111111011011",
    "111111111101111111011011110110111101101111011011",
    "111111111101111111011111110110111101101111011011",
    "111111111101111111011111110110111101101111011011",
    "111111111101111111011111110110111101101111011011",
    "111111111101111111011111110110111101101111011011",
    "111111111101111111011111110110111101101111011011",
    "111111111101111111011111110110111101101110011011",
    "111111111101111111011011110010111101101111011011",
    "111111111100111111000011110000111100011111000011",
    "111111111100111111000011110000111100011111000011",
    "110011111100001111000011110000011100001111000011",
    "111111111111111111111111111111111111111111111111",
    "111111111111111111111111111111111111111111111111",
    "111111111111111111111111110110111111111111111111",
    "111111111111111111011011110110111111111111111111",
    "111111111111111111111011110110111111111111111111",
    "111111111111111100011000000110001111111111111111",
    "110000000000000000011100000110000000001100011111",
    "100001110000001100010001001110010011001100100011",
    "000111000000100000000000000110000001000000010000",
    "000110000001000000000000000110000000100000001000",
    "000110000000000000000000000110000001000000010000",
    "000110000000000000011000000100000001000000000000",
    "000110000000000000011000000100000001000000000000",
    "000000000000000000011000000010000000100000000000",
    "000000000000000000001000000010000000100000000000",
    "000000000000000000011000000000000001000000000000",
    "000000000000100000011100000110000001100000011000",
    "000000000001110000001110000010000000100000001000",
    "000111110000111100001111001111010001110000011100",
    "000111110000111100011111000111010001110000011100",
    "000111110001111100001111001111010001110000011100",
    "000000110000001101000011111111110100001100000011",
    "000100000011100000111111001111100011100000010000",
    "000011100000001100010001000100000000000000000000",
    "000000000000000000010000000100000000000000000000",
    "000000000000000000010000000100000001000000000000",
    "000000000000000000010000000100000001000000010000",
    "000000000000000000001000000110000001100000011000",
    "000000000000100000001000000010000000100000001000",
    "000000000000100000001000000010000000110000001100",
    "000000000001000000011000000110000001100000000000",
    "000000000001000000010000000100000011000000111000",
    "000000000001000000010000000100000001000000111000",
    "000000000001100000011000000110000001110000011100",
    "000000000001110000011000000110000001110000011100",
    "000000000001110000001000000110000001110000011100",
    "000000000001100010111000001110000111100000011000",
    "000000000001100000011000101110000101110000011100",
    "000000000000000000000000000000000000000000000000",
    "000000000001100000011000000110000011100000011000",
    "000000000001100000011000000110000011100000111000",
    "000000000000001100000001000000010000001100000011",
    "000000000011100000011000000110000011100000111000",
    "000000000111000000110000001100000111000011110000",
    "110000111110001111000011110000011000000011000001",
    "110000111110011111000011110000011000000110000011",
    "110001111100011111000011110000111000001111000011",
    "110001111100011111000111110001111110011111100111",
    "000011111100011111000011110000111100101111001111",
    "101111111011111110000011000000010010001100110011",
    "110111111100111111000111000000011001000010010000",
    "111111111110011111110011110100001100000011100100",
    "111101111111001111111001111000001110001011110010",
    "111101111111001111110001111000001110001011110010",
    "111101111111001111110001111000001110001011110010",
    "111111111111001111111001111100001110001011110010",
    "111110111111101111111001111000001110001011110010",
    "111110111111001111111001111000001100000011110010",
    "111110111111100111111001111000001110000011010010",
    "111111001111100011111100111011001110100011110000",
    "111111111111111111111110111001101110011111110111",
    "111111111111111111111111111111111111111111111111",
    "111111111111111111111111111111111111111111111111",
    "111111111111111111111111111111111111111111111111",
    "111111111111111111101111111101111111100111111100",
    "111111111111111111100000111100001111100011111000",
    "111111111110001111100111111001111110011111000111",
    "111111111110011111100111110001111000001110100011",
    "111111111110011111000111000011111100011111100111",
    "111111111001001110000011110001111110011111100011",
    "111111111001101111000011110101111110001111100011",
    "111111111101101111000011110100111110001111100011",
    "111111111101111111000011110100111110001111100011",
    "111111111101111111000011110100111110001111100011",
    "111111111101111111000011110100111110001111100011",
    "111111111101110111000011111100111110001111100011",
    "111111111101111111000011111100111110001111100011",
    "110111111100111111010001110100111110001111100011",
    "110011111110111111110001111100111111001111100011",
    "111011111110111111110011111000111110011111100111",
    "111011111110111111110011111000111110011111100111",
    "111001111100011111110001111000011111001111100011",
    "110001111100011111011000111000001111000011100011",
    "100001111001011110011000110100001110000011100000",
    "011001110111111001111000001100000001000010000000",
    "111101111111111111111111111111111111111011111100",
    "111111111111111111111111111101111111111111111111",
    "111111111111111111111111111111111111111111111111",
    "111111111111111111111111111111111111111111111111",
    "111111111111111111111111111111111111111111111111",
    "111111111111111111111111111111111111111111111111",
    "111111111111111111111111111111111110111111111111",
    "111111111111111111110111111101111111011111110111",
    "111111111111011111110111111001111111001111110011",
    "111111111111011111110111111001111110001111110011",
    "111111111111011111110111111001111110011111110011",
    "111111111111001111110011111101111110011111110011",
    "111110111111101111111011111100111100011111000011",
    "111110011111100111111011111100111101001111000011",
    "111110011111100111111001111100111101001111000011",
    "111110011111100111111001111110011110000111101001",
    "111111001111110011111101111010001110010011110000",
    "111111101111110011101110111100001111100011111000",
    "111111111110111111110001111110011111111111111111",
    "111111111111111111111111111111111110001111111111",
    "111111111111111111111111111011111111011111111111",
    "111111111111111111101111111001111111011111111111",
    "111111111110111111101111111101111111011111111111",
    "111111111110111111101111111001111110011111111111",
    "111111111110011111110111111001111110011111111111",
    "111101111111011111110111111101111110001111100011",
    "111111111111011111110011111000011111001111110001",
    "111111111111011111110011111000111111100111110001",
    "111111111111101111111001111100001111100011110001",
    "111111001111110011001110111100001111111011111110",
    "110111111110011111111000111111111111111111111111",
    "111111111111111111100001111111001111111111111111",
    "111111111111111111111111111111111111111111111111",
    "110000001110011111111111111011111111111111111111",
    "111001111110111111101111111011111111111111111111",
    "111011111110111111101111111111111111111111111111",
    "111011111110111111101111111011111111111111111111",
    "111111111110111111101111000000000000000000010000",
    "110001110000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000010000000100000001000000010000000",
    "000000000110000011100000010000000100000011000000",
    "001000000111000001110000001000000110000001100000",
    "001100000111000001100000001000000110000001100000",
    "001000000110000001100000001100000010000001110000",
    "001000000110000001100000000100000011000001110000",
    "000000000111000001110000000100000011000001110000",
    "000100000011000000010000000100000011000000110000",
    "000000000001110000011100000100000001000000110000",
    "000000000000000000011100000111000001100000111000",
    "000000000000100000001000000010000001100000111000",
    "000000000000100000001000000010000001100000111000",
    "000000000000100000001000000010000001100000011000",
    "000000000000100000001000000010000000100000011000",
    "000000000001100001111000000010000000100000011100",
    "000000000000010000000110000001100000001000000010",
    "000000000000000000000000011110000000000000000000",
    "000000000000000000000000011111110000000000000000",
    "000010010001100100010000000000000011111100000000",
    "000010110000101100010110001101101111111000000111",
    "000111110000101000001100000101001111011000000000",
    "000001100000011000000011000010110000001100000011",
    "000000110000011010000111100001111000011010000110",
    "000000000000001001000110100001101000011000000100",
    "000000000000110001001100010011100100100001001000",
    "000000000001100010011000100011001000100000011000",
    "000000000001100010011000100011001000100010011000",
    "000000000001100000011000000111000000100000011000",
    "000000000001100000110000000110000001111000011000",
    "000100000011000000110000000110000001110000011100",
    "001100000011000001110000001110000001110000011100",
    "001100000011000001110000001110000001110000011100",
    "000100000011000000110000000110000001110000011110",
    "000110000001100000111000000110000001110000011110",
    "000011000001110000011000000011000000111100001111",
    "000001101101111000000110000001110000011100000110",
    "110000111000001110000011100000111000001110000011",
    "110000001100000010000000100000001000000010000001",
    "110000001100000011000000010000000100000000000000",
    "011000000010000000100000000000000000000000000000",
    "000100000001000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000110000001100000000000",
    "000000000000000000000000000000000010010000000000",
    "000000000000000000000000000000000000000000011000",
    "000000000000000000000000000000000001000000000000",
    "000000000000000000000000000000000000000000011000",
    "000000000000000000000000000000000001100000011000",
    "000000000000000000000000000000000001100000011000",
    "000000000000000000000000000000000001100000011000",
    "000000000000000000000000000100000001100000011000",
    "000000000000000000000000000110000001100000011000",
    "000000000000000000010000000110000001100000011000",
    "000000000000000000010000000110000001100000011000",
    "000000000000000000010000000110000011100000011000",
    "000000000001000000010000000110000011100000011000",
    "000000000001000000010000000110000011100000111000",
    "000000000001000000011000000110000001100000111000",
    "000000000000110000001100000011100000111100001111",
    "000000000001100000011000000110000001100100011010",
    "001001000011110000110100001111000011110000111100",
    "011101100111011001110110011111100111111001111110",
    "111101111111011111100111111101111111111111111111",
    "111000111111001111100011111100111111111111111111",
    "111000111110001111100010111100101111111111111111",
    "110111101100111011001110111111101111111011111110",
    "101110001011100010111000111110001111100111111001",
    "101100001011000011110000111100001111000111110001",
    "101100001111000011110000111100001111000111110001",
    "101100001111000011110000111100001111001111110000",
    "101100001111000011110000111100001111001111110000",
    "101100001111000011110000111100001111001111110001",
    "001100001111000011110000111100001111001111110001",
    "011100000111000001110000111100001111000111110001",
    "111100000111000001110000111100001111001111110001",
    "011110000111100011110000111100001110000111100011",
    "000000100001101011111000111000001010011110111111",
    "110000111100001111110000111100000111100000111100",
    "110000111111000111010001011100000111000000111000",
    "011111010111110111111000111000000100000101000001",
    "000111100000111100001111000010110000111110000111",
    "100000000000000001100000111100011111111111111110",
    "011111011111110011111100110000001100000001000001",
    "000001110000011100000111000011110001111100011111",
    "111000001111000011110000111100001111000011111000",
    "111111111011111100001101000011110000000000000000",
    "000010010000000100000011001111110111111101101111",
    "000000001110000011110000101100001111100011111111",
    "111100001111000010110010101100101111000011110000",
    "111000001110000001100100011001001110000011100000",
    "110000001100000011001100110001101100000011000000",
    "100000001000000010001000100011001000000010000000",
    "110000001100000011000000110001001100000011000000",
    "110000001100000011000000110001101100000011000000",
    "111110001111100011011000110110001111100011111000",
    "111111111111111111100111111001110000111100001111",
    "000111110001111100111111000110110000011100000111",
    "111111111100111111001111110010111110001111000011",
    "111111111110111111001111111011111110011111100011",
    "111001111100011111100111111000111110011111100111",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000"},
    fps = 3
}

-- Assuming 'level' object is already defined in your Lua script
local bx = level.left
local by = level.top
local receiveChannel = (videoData.width * videoData.height) + 1

local function relay(receive, send)
    local b = level:placeRelay(bx, by)
    b:setReceivingChannel(receive)
    b:setSendingChannel(send)
    b:setSwitchRequirements("Any Active")
    bx = bx + 1
    if bx > level.right then
        bx = 1
        by = by + 1
    end
end

-- Decode video data and generate relays for each frame
local previousFrame = nil
local frameIndex = 0
for _, frameData in ipairs(videoData.data) do
    if frameData ~= previousFrame then
        for pixel in frameData:gmatch(".") do
            if pixel == "1" then
                relay(receiveChannel, frameIndex)
            end
            frameIndex = frameIndex + 1
        end
        previousFrame = frameData
    else
        -- Place relay with sending channel as the duplicate frame's receive channel
        relay(receiveChannel, receiveChannel - 1)
    end
    receiveChannel = receiveChannel + 1  -- Increment receive channel for the next frame
    frameIndex = 0  -- Reset frame index for the next frame
end
