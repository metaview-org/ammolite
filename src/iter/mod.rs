use std::mem;
use std::marker::PhantomData;
use generic_array::{GenericArray, ArrayLength};
use safe_transmute::PodTransmutable;
use gltf::accessor::Accessor;

pub struct ArrayIterator<T: Clone, U: ArrayLength<T>> {
    next_index: usize,
    data: GenericArray<T, U>,
}

impl<T: Clone, U: ArrayLength<T>> ArrayIterator<T, U> {
    pub fn new<A: Into<GenericArray<T, U>>>(array: A) -> Self {
        Self {
            next_index: 0,
            data: array.into(),
        }
    }
}

impl<T: Clone, U: ArrayLength<T>> Iterator for ArrayIterator<T, U> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len() == 0 {
            None
        } else {
            let result = self.data[self.next_index].clone();

            self.next_index += 1;

            Some(result)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<T: Clone, U: ArrayLength<T>> ExactSizeIterator for ArrayIterator<T, U> {
    fn len(&self) -> usize {
        self.data.len() - self.next_index
    }
}

pub struct ForcedExactSizeIterator<T: Default, I: Iterator<Item=T>> {
    iterator: I,
    forced_remaining_len: usize,
}

impl<T: Default, I: Iterator<Item=T>> ForcedExactSizeIterator<T, I> {
    pub fn new(iterator: I, forced_len: usize) -> Self {
        Self {
            iterator,
            forced_remaining_len: forced_len,
        }
    }
}

impl<T: Default, I: Iterator<Item=T>> Iterator for ForcedExactSizeIterator<T, I> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.forced_remaining_len <= 0 {
            None
        } else {
            let result = self.iterator.next();

            self.forced_remaining_len -= 1;

            result.or_else(|| Some(Self::Item::default()))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.forced_remaining_len, Some(self.forced_remaining_len))
    }
}

impl<T: Default, I: Iterator<Item=T>> ExactSizeIterator for ForcedExactSizeIterator<T, I> {
    fn len(&self) -> usize {
        self.forced_remaining_len
    }
}

pub struct ByteBufferIterator<'a, T: PodTransmutable> {
    next_item_index: usize,
    /// Byte buffer with elements starting at index 0 (offset applied)
    bytes: &'a [u8],
    /// The number of bytes between the starting byte of each element
    stride: usize,
    /// Number of elements of type `T` to read from the `bytes` slice
    items: usize,
    _phantom_data: PhantomData<T>,
}

impl<'a, T: PodTransmutable> ByteBufferIterator<'a, T> {
    pub fn new(bytes: &'a [u8], stride: usize, items: usize) -> Self {
        assert!(items == 0 || (items - 1) * stride + mem::size_of::<T>() <= bytes.len());

        Self {
            next_item_index: 0,
            bytes,
            stride,
            items,
            _phantom_data: PhantomData,
        }
    }

    pub fn from_accessor(
        buffer_data_array: &'a [gltf::buffer::Data],
        accessor: &Accessor,
    ) -> Self {
        let view = accessor.view();
        let stride = view.stride().unwrap_or_else(|| accessor.size());
        let slice_offset = view.offset() + accessor.offset();
        let slice_len = stride * accessor.count();
        let slice: &[u8] = &buffer_data_array[view.buffer().index()][slice_offset..(slice_offset + slice_len)];

        Self::new(slice, stride, accessor.count())
    }
}

impl<'a, T: PodTransmutable> Iterator for ByteBufferIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_item_index >= self.items {
            None
        } else {
            let item_slice_start_index = self.next_item_index * self.stride;
            let item_slice_range = item_slice_start_index..(item_slice_start_index + mem::size_of::<T>());
            let item_slice = &self.bytes[item_slice_range];
            self.next_item_index += 1;

            Some(safe_transmute::guarded_transmute_pod::<T>(item_slice).unwrap())
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, T: PodTransmutable> ExactSizeIterator for ByteBufferIterator<'a, T> {
    fn len(&self) -> usize {
        self.items - self.next_item_index
    }
}
