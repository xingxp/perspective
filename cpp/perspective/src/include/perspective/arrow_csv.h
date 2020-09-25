/******************************************************************************
 *
 * Copyright (c) 2017, the Perspective Authors.
 *
 * This file is part of the Perspective library, distributed under the terms of
 * the Apache License 2.0.  The full license can be found in the LICENSE file.
 *
 */

#pragma once
#include <unordered_map>
#include <arrow/io/memory.h>
#include <arrow/table.h>

namespace perspective {
namespace apachearrow {




    /**
     * @brief Initialize the arrow loader with a CSV.
     * 
     * @param ptr 
     */
    std::shared_ptr<::arrow::Table> csvToTable(std::string& csv, bool is_update, std::unordered_map<std::string, std::shared_ptr<arrow::DataType>>& schema);


} // namespace arrow
} // namespace perspective