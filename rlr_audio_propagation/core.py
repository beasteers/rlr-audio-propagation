import numpy as np
import _rlr_audio_propagation as _r

# W, Z, X, Y -> X, Y, Z
AMBISONICS_XYZ_INDEX = [3, 1, 2]


class Context(_r.Context):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def reset(self, config=None):
        if config is None:
            config = self.config  # XXX: is this correct?
        super().reset(config)
        self.config = config

    @property
    def sr(self):
        """Audio sample rate."""
        return self.config.sampleRate
    
    def get_channel_counts(self):
        """Get the number of channels for each listener-source combination."""
        return [
            [
                self.get_ir_channel_count(i, j)
                for j in range(self.get_source_count())
            ]
            for i in range(self.get_listener_count())
        ]
    
    def get_ir_channel(self, listener_index, source_index, channel_index):
        """Get audio for a specific listener-source-channel combination."""
        return np.array(super().get_ir_channel(listener_index, source_index, channel_index))
    

    def get_listener_source_channel_audio(self, listener_index, source_index, channel_index):
        """Get audio for a specific listener-source-channel combination."""
        return self.get_ir_channel(listener_index, source_index, channel_index)

    def get_listener_source_audio(self, listener_index, source_index):
        """Get audio for a specific listener-source combination for all channels."""
        return padded_stack([
            self.get_listener_source_channel_audio(listener_index, source_index, k)
            for k in range(self.get_ir_channel_count(listener_index, source_index))
        ])
    
    def get_listener_audio(self, listener_index):
        """Get audio for a specific listener for all sources and channels."""
        return padded_stack([
            [
                self.get_listener_source_channel_audio(listener_index, j, k)
                for k in range(self.get_ir_channel_count(listener_index, j))
            ] 
            for j in range(self.get_source_count())
        ])
    
    def get_source_audio(self, source_index):
        """Get audio for a specific source for all listeners and channels."""
        return padded_stack([
            [
                self.get_listener_source_channel_audio(i, source_index, j)
                for j in range(self.get_ir_channel_count(i, source_index))
            ] 
            for i in range(self.get_listener_count())
        ])
    
    def get_audio(self):
        """Get audio for all listeners, sources, and channels."""
        return padded_stack([
            [
                [
                    self.get_listener_source_channel_audio(i, j, k)
                    for k in range(self.get_ir_channel_count(i, j))
                ] 
                for j in range(self.get_source_count())
            ] 
            for i in range(self.get_listener_count())
        ])


def maxlen_recursive(arrays):
    """Get the maximum length of a nested list of arrays."""
    return max((maxlen_recursive(c) for c in arrays), default=0) if isinstance(arrays, list) else len(arrays)


def pad_recursive(channel, max_length):
    """Pad a nested list of arrays to a maximum length."""
    if isinstance(channel, list):
        return [pad_recursive(c, max_length) for c in channel]
    return np.pad(channel, (0, max_length - len(channel)))


def padded_stack(channels):
    """Stack arrays of different lengths, padding with zeros."""
    return np.array(pad_recursive(channels, maxlen_recursive(channels)))