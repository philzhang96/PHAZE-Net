def chunk_sequence(sequence, window_size, stride):
    """
    Splits a list into overlapping temporal chunks.
    
    Parameters:
        sequence (List): The input list of ROIs or features per frame.
        window_size (int): Number of frames in each chunk.
        stride (int): Step size between chunks.
    
    Returns:
        List[List]: Chunks (each a list of frames).
    """
    chunks = []
    for i in range(0, len(sequence) - window_size + 1, stride):
        chunk = sequence[i:i + window_size]
        chunks.append(chunk)
    return chunks


def multi_resolution_chunking(roi_dict, config):
    """
    Apply multi-scale chunking across all ROI regions.

    Parameters:
        roi_dict (dict): {region_name: [frame_0_crop, frame_1_crop, ...]}
        config (dict): Defines chunk sizes and strides for each scale.

    Returns:
        dict: {region_name: {scale: [chunks]}}
    """
    chunked = {}
    for region, frames in roi_dict.items():
        region_chunks = {}
        for scale, params in config.items():
            window_size = params["size"]
            stride = params["stride"]
            region_chunks[scale] = chunk_sequence(frames, window_size, stride)
        chunked[region] = region_chunks
    return chunked
