import shapelets.compute as sc

def test_norm_z():
    data = sc.array([0,1,2,3,4,5,6,7], (4,2), dtype="float32")
    result = sc.normalization.zscore(data)
    expected = sc.array([
        -1.341640786499870, -0.447213595499958, 0.447213595499958, 1.341640786499870,
        -1.341640786499870, -0.447213595499958, 0.447213595499958, 1.341640786499870], (4,2))

    assert result.shape == data.shape
    assert expected.same_as(result)

def test_norm_z_lowstd():
    data = sc.zeros((100,1), dtype="float32")    
    data[49] = -0.0001
    data[50] = 0.0001
    result = sc.normalization.zscore(data)
    assert result[48] == 0.0 
    assert result[49] == result[50]
    assert result[51] == 0.0 
