// (c) Copyright 2024 Fulcrum Technologies, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use vortex::array::{Array, ArrayRef};

use crate::REEArray;

mod private {
    pub trait Sealed {}
}

pub trait DowncastREE: private::Sealed {
    fn maybe_ree(&self) -> Option<&REEArray>;

    fn as_ree(&self) -> &REEArray {
        self.maybe_ree().unwrap()
    }
}

impl private::Sealed for dyn Array {}

impl DowncastREE for dyn Array {
    fn maybe_ree(&self) -> Option<&REEArray> {
        self.as_any().downcast_ref()
    }
}

impl private::Sealed for ArrayRef {}

impl DowncastREE for ArrayRef {
    fn maybe_ree(&self) -> Option<&REEArray> {
        self.as_any().downcast_ref()
    }
}