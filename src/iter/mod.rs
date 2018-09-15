use generic_array::{GenericArray, ArrayLength};

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

